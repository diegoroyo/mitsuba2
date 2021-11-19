#include <enoki/stl.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/transientintegrator.h>
#include <random>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class TransientPathIntegrator : public TransientMonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(TransientMonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Texture, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    TransientPathIntegrator(const Properties &props) : Base(props) {
        m_filter_depth = props.int_("filter_depth", -1);
        // avoid the case filter_depth >= max_depth
        Assert(m_filter_depth < m_max_depth);
        m_discard_direct_paths = props.bool_("discard_direct_paths", false);
        // avoid the case m_discard_direct_paths && m_filter_depth > 0
        Assert(!m_discard_direct_paths || m_filter_depth <= 0);
        m_nlos_emitter_sampling = props.bool_("nlos_emitter_sampling", false);
        surface_light = nullptr;
    }

    void sample(const Scene *scene, Sampler *sampler, const RayDifferential3f &ray_,
                const Medium * /* medium */,
                std::vector<FloatTimeSample<Float, Mask>> & /* aovs_record */,
                std::vector<RadianceSample<Float, Spectrum, Mask>> &timed_samples_record,
                Float max_path_opl,
                Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        RayDifferential3f ray = ray_;

        // Index of refraction of the current medium (used for OPL calculation)
        Float current_ior(lookup_ior("air"));

        // MIS weight for intersected emitters (set by prev. iteration)
        Float emission_weight(1.f);

        Spectrum throughput(1.f);

        // Time associated to a path, measured in optical path length
        // NOTE(diego): this assumes that the ray's `time` variable is measured
        // in OPL, we'll just have to believe now :-)
        Float path_opl(ray.time);

        Point3f nlos_laser_target;
        if (m_nlos_emitter_sampling) {
            auto emitters = scene->emitters();
            if (unlikely(emitters.size() != 1)) {
                Throw("NLOS emitter sampling is not implemented for scenes "
                      "with more than one emitter.");
            }
            Transform4f trafo =
                emitters[0]->world_transform()->eval(ray.time, true);
            Vector3f laser_dir = trafo.transform_affine(Vector3f(0, 0, 1));
            Ray3f ray_laser(trafo.translation(), laser_dir, ray.time);
            nlos_laser_target = scene->ray_intersect(ray_laser, true).p;
        }

        // ---------------------- First intersection ----------------------

        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        EmitterPtr emitter = si.emitter(scene);

        for (int depth = 1;; ++depth) {

            // FIXME(diego): medium index of refraction should be used to scale
            // the distance in order to obtain OPL.
            path_opl += si.distance(ray) * current_ior;

            // Transient samples are not going to get stored anyway, discard
            if (path_opl > max_path_opl)
                break;

            // ---------------- Intersection with emitters ----------------

            if (any_or<true>(neq(emitter, nullptr))) {
                Spectrum radiance(0.f);
                radiance[active] += emission_weight * throughput * emitter->eval(si, active);
                Mask path_finished = active;
                if (depth == m_filter_depth || (!m_discard_direct_paths || depth >= 2))
                    timed_samples_record.emplace_back(path_opl, radiance, path_finished);
            }

            active &= si.is_valid();

            // Stop if we've exceeded the number of requested bounces, or
            // if there are no more active lanes. Only do this latter check
            // in GPU mode when the number of requested bounces is infinite
            // since it causes a costly synchronization.
            if ((uint32_t) depth >= (uint32_t) m_max_depth ||
                ((!is_cuda_array_v<Float> || m_max_depth < 0) && none(active)))
                break;

            // --------------------- Emitter sampling ---------------------

            BSDFContext ctx;
            BSDFPtr bsdf  = si.bsdf(ray);
            Mask active_e = active && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            if (likely(any_or<true>(active_e))) {
                const auto f_emitter_sample =
                    [*this](const Scene *scene, Sampler *sampler,
                            BSDFContext &ctx, SurfaceInteraction3f &si,
                            Mask &active_e,
                            std::vector<RadianceSample<Float, Spectrum, Mask>>
                                &timed_samples_record,
                            const BSDFPtr &bsdf, const Spectrum &throughput,
                            const Float &path_opl, const Float &current_ior,
                            const int &depth) {

                    DirectionSample3f ds;
                    Spectrum emitter_val;
                    if(m_nlos_emitter_sampling){
                        surface_emitter(scene);     // Set-up the surface emitter

                        // Fast path to the emitter
                        std::tie(ds, emitter_val) = 
                            surface_light->sample_direction(si,
                                            sampler->next_2d(active_e), active_e);
                        active_e &= neq(ds.pdf, 0.f);

                        // Perform a visibility test if requested
                        if (any_or<true>(active_e)) {
                            Ray3f ray(si.p, ds.d, 
                                    math::RayEpsilon<Float> * (1.f + hmax(abs(si.p))),
                                    ds.dist * (1.f - math::ShadowEpsilon<Float>),
                                    si.time, si.wavelengths);
                            emitter_val[scene->ray_test(ray, active_e)] = 0.f;
                        }
                    } else
                        std::tie(ds, emitter_val) =
                            scene->sample_emitter_direction(
                                si, sampler->next_2d(active_e), true, active_e);
                    
                    active_e &= neq(ds.pdf, 0.f);

                    // Query the BSDF for that emitter-sampled direction
                    Vector3f wo       = si.to_local(ds.d);
                    Spectrum bsdf_val = select(
                        neq(bsdf, nullptr),
                        si.to_world_mueller(
                            bsdf->eval(ctx, si, wo, active_e), -wo, si.wi),
                        0.0f);

                    // Determine density of sampling that same direction
                    // using BSDF sampling
                    Float bsdf_pdf =
                        select(neq(bsdf, nullptr),
                                bsdf->pdf(ctx, si, wo, active_e), 0.0f);

                    Float mis =
                        select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                    Spectrum radiance(0.f);
                    radiance[active_e] +=
                        mis * throughput * bsdf_val * emitter_val;

                    if (depth == m_filter_depth || (!m_discard_direct_paths || depth >= 2))
                        timed_samples_record.emplace_back(
                            path_opl + ds.dist * current_ior, radiance,
                            active_e);
                };

				// same emitter sampling as before
				f_emitter_sample(scene, sampler, ctx, si, active_e,
								 timed_samples_record, bsdf, throughput,
								 path_opl, current_ior, depth);
            }

            // ----------------------- BSDF sampling ----------------------

            // Sample BSDF * cos(theta)
            auto [bs, bsdf_val] =
                bsdf->sample(ctx, si, sampler->next_1d(active), sampler->next_2d(active), active);
            bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

            throughput = throughput * bsdf_val;
            active &= any(neq(depolarize(throughput), 0.f));
            if (none_or<false>(active))
                break;

            current_ior = bs.eta;

            // Intersect the BSDF ray against the scene geometry
            ray                          = si.spawn_ray(si.to_world(bs.wo));
            SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray, active);

            /* Determine probability of having sampled that same
               direction using emitter sampling. */
            // NOTE(diego): NLOS scenes (should) only have point light emitters,
            //              so they cannot be reached with BSDF sampling
            emitter = si_bsdf.emitter(scene, active);
            DirectionSample3f ds(si_bsdf, si);
            ds.object = emitter;

            if (any_or<true>(neq(emitter, nullptr))) {
                Float emitter_pdf =
                    select(neq(emitter, nullptr) && !has_flag(bs.sampled_type, BSDFFlags::Delta),
                           scene->pdf_emitter_direction(si, ds), 0.f);

                emission_weight = mis_weight(bs.pdf, emitter_pdf);
            }

            si = std::move(si_bsdf);
        }
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("TransientPathIntegrator[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    MTS_DECLARE_CLASS()
private:
    // Return the laser-projection emitter of the scene into a pointlight in the
    // relly surface
    EmitterPtr surface_emitter(const Scene *scene){
        if(!surface_light){
            auto emitters = scene->emitters();
            if (unlikely(emitters.size() != 1)) {
                Throw("NLOS emitter sampling is not implemented for scenes "
                        "with more than one emitter.");
            }

            Float time = 0.0; 
            Transform4f trafo =
                emitters[0]->world_transform()->eval(time, true);
            Vector3f laser_dir = trafo.transform_affine(Vector3f(0, 0, 1));
            Ray3f ray_laser(trafo.translation(), laser_dir, time);
            // Laser intersection to surface
            SurfaceInteraction si_laser_target = scene->ray_intersect(ray_laser, true);

            // Point in the surface (really close to be reached)
            Vector3f point_light_pos = si_laser_target.p - si_laser_target.n*0.001;
            // Point light properties
            Properties pl_props("point");
            pl_props.set_array3f("position", point_light_pos);
            pl_props.texture<Texture>("intensity", Texture::D65(1.0f));

            // Returns a pointlight
            auto pmgr = PluginManager::instance();
            surface_light = 
                static_cast<Emitter *>( pmgr->create_object<Emitter>(pl_props) );
        }
        return surface_light;
    }

    EmitterPtr surface_emitter(const Scene *scene) const{
        if(!surface_light){
            auto emitters = scene->emitters();
            if (unlikely(emitters.size() != 1)) {
                Throw("NLOS emitter sampling is not implemented for scenes "
                        "with more than one emitter.");
            }

            Float time = 0.0; 
            Transform4f trafo =
                emitters[0]->world_transform()->eval(time, true);
            Vector3f laser_dir = trafo.transform_affine(Vector3f(0, 0, 1));
            Ray3f ray_laser(trafo.translation(), laser_dir, time);
            // Laser intersection to surface
            SurfaceInteraction si_laser_target = scene->ray_intersect(ray_laser, true);

            // Point in the surface (really close to be reached)
            Vector3f point_light_pos = si_laser_target.p - si_laser_target.n*0.001;
            // Point light properties
            Properties pl_props("point");
            pl_props.set_array3f("position", point_light_pos);
            pl_props.texture<Texture>("intensity", Texture::D65(1.0f));

            // Returns a pointlight
            auto pmgr = PluginManager::instance();
            return
                static_cast<Emitter *>( pmgr->create_object<Emitter>(pl_props) );
        }
        return surface_light;
    }


    int m_filter_depth;
    bool m_discard_direct_paths;
    bool m_nlos_emitter_sampling;
    EmitterPtr surface_light;
};

MTS_IMPLEMENT_CLASS_VARIANT(TransientPathIntegrator, TransientMonteCarloIntegrator)
MTS_EXPORT_PLUGIN(TransientPathIntegrator, "Transient Path Tracer integrator");
NAMESPACE_END(mitsuba)
