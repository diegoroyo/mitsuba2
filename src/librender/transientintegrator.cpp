#include <thread>
#include <mutex>

#include <enoki/morton.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/streakfilm.h>
#include <mitsuba/render/transientintegrator.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/spiral.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT TransientIntegrator<Float, Spectrum>::TransientIntegrator(const Properties &props)
    : Base(props) {}

MTS_VARIANT TransientIntegrator<Float, Spectrum>::~TransientIntegrator() = default;

// -----------------------------------------------------------------------------

MTS_VARIANT TransientSamplingIntegrator<Float, Spectrum>::TransientSamplingIntegrator(const Properties &props)
    : Base(props) {

    m_block_size = (uint32_t) props.size_("block_size", 0);
    uint32_t block_size = math::round_to_power_of_two(m_block_size);
    if (m_block_size > 0 && block_size != m_block_size) {
        Log(Warn, "Setting block size from %i to next higher power of two: %i", m_block_size,
            block_size);
        m_block_size = block_size;
    }

    m_samples_per_pass = (uint32_t) props.size_("samples_per_pass", (uint32_t) -1);
    m_timeout = props.float_("timeout", -1.f);

    /// Disable direct visibility of emitters if needed
    m_hide_emitters = props.bool_("hide_emitters", false);
}

MTS_VARIANT TransientSamplingIntegrator<Float, Spectrum>::~TransientSamplingIntegrator() = default;

MTS_VARIANT void TransientSamplingIntegrator<Float, Spectrum>::cancel() {
    m_stop = true;
}

MTS_VARIANT std::vector<std::string> TransientSamplingIntegrator<Float, Spectrum>::aov_names() const {
    return { };
}

MTS_VARIANT bool TransientSamplingIntegrator<Float, Spectrum>::render(Scene *scene, Sensor *sensor) {
    ScopedPhase sp(ProfilerPhase::Render);
    m_stop = false;

    ref<StreakFilm> film = dynamic_cast<StreakFilm *>(sensor->film());
    ScalarVector2i film_size = film->crop_size();

    uint32_t total_spp = sensor->sampler()->sample_count();
    uint32_t samples_per_pass =
        (m_samples_per_pass == (uint32_t) -1)
            ? total_spp
            : std::min((uint32_t) m_samples_per_pass, total_spp);
    if ((total_spp % samples_per_pass) != 0)
        Throw("sample_count (%d) must be a multiple of samples_per_pass (%d).",
              total_spp, samples_per_pass);

    size_t n_passes = (total_spp + samples_per_pass - 1) / samples_per_pass;

    std::vector<std::string> channels = aov_names();
    bool has_aovs = !channels.empty();

    // Insert default channels and set up the film
    uint channels_to_push = sensor->is_nlos_sensor() ? 3 : 5;
    for (size_t i = 0; i < channels_to_push; ++i)
        channels.insert(channels.begin() + i, std::string(1, "XYZAW"[i]));
    if (film->should_auto_detect_bins())
        film->auto_detect_bins(scene, sensor);
    film->prepare(channels);

    prepare_integrator(scene);
    m_render_timer.reset();
    if constexpr (!is_cuda_array_v<Float>) {
        /// Render on the CPU using a spiral pattern
        size_t n_threads = __global_thread_count;
        Log(Info, "Starting render job (%ix%i, %i sample%s,%s %i thread%s)",
            film_size.x(), film_size.y(),
            total_spp, total_spp == 1 ? "" : "s",
            n_passes > 1 ? tfm::format(" %d passes,", n_passes) : "",
            n_threads, n_threads == 1 ? "" : "s");

        if (m_timeout > 0.f)
            Log(Info, "Timeout specified: %.2f seconds.", m_timeout);

        // Find a good block size to use for splitting up the total workload.
        if (m_block_size == 0) {
            uint32_t block_size = MTS_BLOCK_SIZE;
            while (true) {
                if (block_size == 1 || hprod((film_size + block_size - 1) / block_size) >= n_threads)
                    break;
                block_size /= 2;
            }
            m_block_size = block_size;
        }

        Spiral spiral(film, m_block_size, n_passes);

        ThreadEnvironment env;
        ref<ProgressReporter> progress = new ProgressReporter("Rendering");
        std::mutex mutex;

        // Total number of blocks to be handled, including multiple passes.
        size_t total_blocks = spiral.block_count() * n_passes, blocks_done = 0;

        // each pass/block is divided into multiple divs so progress bar updates
        // more smoothly
        size_t samples_per_div = 100000;
        size_t n_divs          = 1;
        if (samples_per_pass < samples_per_div * 3) {
            Log(Info, "Using 1 div");
        } else {
            n_divs = (samples_per_pass - 1) / samples_per_div + 1;
            Log(Info, "Using (%i - 1) / %i + 1 = %i divs", samples_per_pass,
                samples_per_div, n_divs);
        }

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, total_blocks, 1),
            [&](const tbb::blocked_range<size_t> &range) {
                ScopedSetThreadEnvironment set_env(env);
                ref<Sampler> sampler        = sensor->sampler()->clone();
                ref<StreakImageBlock> block = new StreakImageBlock(
                    m_block_size, film->num_bins(), film->bin_width_opl(),
                    film->start_opl(), channels.size(),
                    film->reconstruction_filter(),
                    film->time_reconstruction_filter(), !has_aovs);
                scoped_flush_denormals flush_denormals(true);

                // For each block
                for (auto i = range.begin(); i != range.end() && !should_stop();
                     ++i) {
                    auto [offset, size, block_id] = spiral.next_block();
                    Assert(hprod(size) != 0);
                    block->set_size(size, film->num_bins());
                    block->set_offset(offset);
                    for (size_t div = 0; div < n_divs && !should_stop();
                         div++) {
                        size_t n_samples =
                            div == n_divs - 1 ? (samples_per_pass -
                                                 (n_divs - 1) * samples_per_div)
                                              : samples_per_div;

                        std::vector<FloatSample<Float>> aovs_record;
                        render_block(scene, sensor, sampler, block, aovs_record,
                                     n_samples, block_id * n_divs + div);

                        film->put(block);

                        /* Critical section: update progress bar */ {
                            std::lock_guard<std::mutex> lock(mutex);
                            blocks_done++;
                            progress->update(
                                blocks_done /
                                (ScalarFloat)(total_blocks * n_divs));
                        }
                    }
                }
            });
    } else {
        Log(Info, "Start rendering...");

        ref<Sampler> sampler = sensor->sampler();
        sampler->set_samples_per_wavefront((uint32_t) samples_per_pass);

        ScalarFloat diff_scale_factor = rsqrt((ScalarFloat) sampler->sample_count());
        ScalarUInt32 wavefront_size = hprod(film_size) * (uint32_t) samples_per_pass;
        if (sampler->wavefront_size() != wavefront_size)
            sampler->seed(0, wavefront_size);

        UInt32 idx = arange<UInt32>(wavefront_size);
        if (samples_per_pass != 1)
            idx /= (uint32_t) samples_per_pass;

        ref<StreakImageBlock> block = new StreakImageBlock(m_block_size,
                                                           film->num_bins(),
                                                           film->bin_width_opl(),
                                                           film->start_opl(),
                                                           channels.size(),
                                                           film->reconstruction_filter(),
                                                           film->time_reconstruction_filter(),
                                                           !has_aovs);
        block->clear();
        block->set_offset(sensor->film()->crop_offset());

        Vector2f pos = Vector2f(Float(idx % uint32_t(film_size[0])),
                                Float(idx / uint32_t(film_size[0])));
        pos += block->offset();

        // TODO(jorge): fix for GPU
        std::vector<Float> aovs(channels.size());

        for (size_t i = 0; i < n_passes; i++)
            render_sample(scene, sensor, sampler, block, aovs.data(),
                          pos, diff_scale_factor);

        film->put(block);
    }

    if (!m_stop)
        Log(Info, "Rendering finished. (took %s)",
            util::time_string(m_render_timer.value(), true));

    return !m_stop;
}

MTS_VARIANT void TransientSamplingIntegrator<Float, Spectrum>::render_block(const Scene *scene,
                                                                   const Sensor *sensor,
                                                                   Sampler *sampler,
                                                                   StreakImageBlock *block,
                                                                   std::vector<FloatSample<Float>> &aovs_record,
                                                                   size_t sample_count_,
                                                                   size_t block_id) const {
    block->clear();
    uint32_t pixel_count  = (uint32_t)(m_block_size * m_block_size),
             sample_count = (uint32_t)(sample_count_ == (size_t) -1
                                           ? sampler->sample_count()
                                           : sample_count_);

    ScalarFloat diff_scale_factor = rsqrt((ScalarFloat) sampler->sample_count());

    if constexpr (!is_array_v<Float>) {
        for (uint32_t i = 0; i < pixel_count && !should_stop(); ++i) {
            sampler->seed(block_id * pixel_count + i);

            ScalarPoint2u pos = enoki::morton_decode<ScalarPoint2u>(i);
            if (any(pos >= block->size()))
                continue;

            pos += block->offset();
            for (uint32_t j = 0; j < sample_count && !should_stop(); ++j) {
                render_sample(scene, sensor, sampler, block, aovs_record,
                              pos, diff_scale_factor);
                aovs_record.clear();
            }
        }
    } else if constexpr (is_array_v<Float> && !is_cuda_array_v<Float>) {
        // Ensure that the sample generation is fully deterministic
        sampler->seed(block_id);

        uint32_t a = pixel_count * sample_count;
        for (auto [index, active] : range<UInt32>(pixel_count * sample_count)) {
            if (should_stop())
                break;
            Float b = index;
            Point2u pos = enoki::morton_decode<Point2u>(index / UInt32(sample_count));
            active &= !any(pos >= block->size());
            pos += block->offset();
            render_sample(scene, sensor, sampler, block, aovs_record, pos, diff_scale_factor, active);
            aovs_record.clear();
        }
    } else {
        ENOKI_MARK_USED(scene);
        ENOKI_MARK_USED(sensor);
        ENOKI_MARK_USED(aovs_record);
        ENOKI_MARK_USED(diff_scale_factor);
        ENOKI_MARK_USED(pixel_count);
        ENOKI_MARK_USED(sample_count);
        Throw("Not implemented for CUDA arrays.");
    }
}

MTS_VARIANT void
TransientSamplingIntegrator<Float, Spectrum>::render_sample(const Scene *scene,
                                                   const Sensor *sensor,
                                                   Sampler *sampler,
                                                   StreakImageBlock *block,
                                                   std::vector<FloatSample<Float>> &aovs_record,
                                                   const Vector2f &pos,
                                                   ScalarFloat diff_scale_factor,
                                                   Mask active) const {
    Vector2f position_sample = pos + sampler->next_2d(active);

    Point2f aperture_sample(.5f);
    if (sensor->needs_aperture_sample())
        aperture_sample = sampler->next_2d(active);

    if (sensor->shutter_open() != 0.f || sensor->shutter_open_time() != 0.f)
        Log(Warn, "Shutter open/close time is not set to zero. Ignoring.");

    Float max_opl = math::Infinity<Float>;
    const auto *streak_film = dynamic_cast<const StreakFilm *>(sensor->film());
    if (streak_film)
        max_opl = streak_film->end_opl();
    else
        Log(Warn, "Cannot determine maximum/end optical path length. Are you "
                  "using a Transient Integrator without a Streak Film?");

    Float wavelength_sample = sampler->next_1d(active);

    Vector2f adjusted_position =
        (position_sample - sensor->film()->crop_offset()) /
        sensor->film()->crop_size();

    auto [ray, ray_weight] = sensor->sample_ray_differential(
        0.f, wavelength_sample, adjusted_position, aperture_sample);

    ray.scale_differential(diff_scale_factor);

    const Medium *medium = sensor->medium();
    std::vector<RadianceSample<Float, Spectrum>> timed_samples_record;
    sample(scene, sampler, ray, medium, aovs_record, timed_samples_record,
           max_opl, active);

    // Either there are no aovs samples because the integrator does not produce
    // aovs and the aov vector is empty or the integrator produces aovs and
    // there are as many aov samples as radiance samples (one aov sample for
    // each radiance sample)
    assert(aovs_record.empty() ||
           (aovs_record.size() == timed_samples_record.size()));
    ENOKI_MARK_USED(ray.wavelengths);
    if (aovs_record.empty()) {
        for (size_t i = 0; i < timed_samples_record.size(); ++i) {
            aovs_record.push_back(FloatSample<Float>(
                timed_samples_record[i].opl, timed_samples_record[i].mask));
        }
    }
    for (size_t i = 0; i < timed_samples_record.size(); ++i) {
        auto &radiance_sample = timed_samples_record[i];
        auto &radiance        = radiance_sample.radiance;
        const auto &mask      = radiance_sample.mask;

        radiance                   = ray_weight * radiance;
        UnpolarizedSpectrum spec_u = depolarize(radiance);
        Color3f xyz;
        if constexpr (is_monochromatic_v<Spectrum>) {
            xyz = spec_u.x();
        } else if constexpr (is_rgb_v<Spectrum>) {
            xyz = srgb_to_xyz(spec_u, mask);
        } else {
            static_assert(is_spectral_v<Spectrum>);
            xyz = spectrum_to_xyz(spec_u, ray.wavelengths, mask);
        }

        if (!sensor->is_nlos_sensor()) {
            // AW pushed backwards
            aovs_record[i].push_front(1.f);
            aovs_record[i].push_front(select(mask, Float(1.f), Float(0.f)));
        }
        // XYZ pushed backwards
        aovs_record[i].push_front(xyz.z());
        aovs_record[i].push_front(xyz.y());
        aovs_record[i].push_front(xyz.x());
    }

    block->put(position_sample, aovs_record);

    sampler->advance();
}

MTS_VARIANT void TransientSamplingIntegrator<Float, Spectrum>::sample(
    const Scene * /* scene */, Sampler * /* sampler */,
    const RayDifferential3f & /* ray */, const Medium * /* medium */,
    std::vector<FloatSample<Float>> & /* aovs_record */,
    std::vector<RadianceSample<Float, Spectrum>> & /* timed_samples_record */,
    Float /* max_path_opl */, Mask /* active */) const {
    NotImplementedError("sample");
}

// -----------------------------------------------------------------------------

MTS_VARIANT TransientMonteCarloIntegrator<Float, Spectrum>::TransientMonteCarloIntegrator(const Properties &props)
    : Base(props) {
    /// Depth to begin using russian roulette
    m_rr_depth = props.int_("rr_depth", 5);
    if (m_rr_depth <= 0)
        Throw("\"rr_depth\" must be set to a value greater than zero!");

    /*  Longest visualized path depth (``-1 = infinite``). A value of \c 1 will
        visualize only directly visible light sources. \c 2 will lead to
        single-bounce (direct-only) illumination, and so on. */
    m_max_depth = props.int_("max_depth", -1);
    if (m_max_depth < 0 && m_max_depth != -1)
        Throw("\"max_depth\" must be set to -1 (infinite) or a value >= 0");
}

MTS_VARIANT TransientMonteCarloIntegrator<Float, Spectrum>::~TransientMonteCarloIntegrator() = default;

MTS_IMPLEMENT_CLASS_VARIANT(TransientIntegrator, Integrator)
MTS_IMPLEMENT_CLASS_VARIANT(TransientSamplingIntegrator, TransientIntegrator)
MTS_IMPLEMENT_CLASS_VARIANT(TransientMonteCarloIntegrator, TransientSamplingIntegrator)

MTS_INSTANTIATE_CLASS(TransientIntegrator)
MTS_INSTANTIATE_CLASS(TransientSamplingIntegrator)
MTS_INSTANTIATE_CLASS(TransientMonteCarloIntegrator)
NAMESPACE_END(mitsuba)
