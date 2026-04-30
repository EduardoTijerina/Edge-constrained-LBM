#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <zmq.hpp>

namespace {

constexpr int NUM_DOF = 29;
constexpr double CONTROL_HZ = 50.0;
constexpr auto FRAME_PERIOD =
    std::chrono::microseconds{static_cast<int64_t>(1'000'000.0 / CONTROL_HZ)};

constexpr const char* TRIGGER_ENDPOINT = "ipc:///tmp/handshake_trigger";
constexpr const char* POLICY_ENDPOINT  = "tcp://127.0.0.1:5556";
constexpr const char* DEFAULT_REF_BIN =
    "/home/eduardot/Documents/Research/Masters research/Resources/"
    "shake_hand05_poses_50hz.bin";

// Arm joint indices in the 29-DoF isaac order (shoulders/elbows/wrists).
// Used to find the frame where the arm is most extended for the HOLDING pose.
constexpr int ARM_JOINTS[] = {11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true, std::memory_order_relaxed); }

enum class TriggerKind { None, Handshake, Release };

TriggerKind classify_trigger(const std::string& s) {
    if (s == "TRIGGER: HANDSHAKE" || s == "hand_extended") return TriggerKind::Handshake;
    if (s == "TRIGGER: RELEASE")                          return TriggerKind::Release;
    return TriggerKind::None;
}

std::vector<float> load_reference(const std::string& path, std::size_t& num_frames) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("cannot open reference bin: " + path);
    }
    const std::streamsize bytes = f.tellg();
    const std::size_t frame_bytes = NUM_DOF * sizeof(float);
    if (bytes <= 0 || static_cast<std::size_t>(bytes) % frame_bytes != 0) {
        throw std::runtime_error(
            "reference file size " + std::to_string(bytes) +
            " is not a multiple of " + std::to_string(frame_bytes));
    }
    num_frames = static_cast<std::size_t>(bytes) / frame_bytes;

    std::vector<float> data(num_frames * NUM_DOF);
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    if (!f) {
        throw std::runtime_error("short read on reference bin: " + path);
    }
    return data;
}

// Find the frame whose arm joints deviate most from the first frame.
// That's the "peak extension" pose we hold while the user keeps presenting.
std::size_t find_peak_extension(const std::vector<float>& ref, std::size_t num_frames) {
    if (num_frames == 0) return 0;
    const float* f0 = ref.data();
    std::size_t best = 0;
    double best_dev = -1.0;
    for (std::size_t i = 0; i < num_frames; ++i) {
        const float* fi = ref.data() + i * NUM_DOF;
        double dev = 0.0;
        for (int j : ARM_JOINTS) {
            double d = static_cast<double>(fi[j]) - static_cast<double>(f0[j]);
            dev += d * d;
        }
        if (dev > best_dev) {
            best_dev = dev;
            best = i;
        }
    }
    return best;
}

void publish_frame(zmq::socket_t& pub, const std::vector<float>& ref, std::size_t idx) {
    constexpr std::size_t frame_bytes = NUM_DOF * sizeof(float);
    zmq::message_t out(frame_bytes);
    std::memcpy(out.data(), ref.data() + idx * NUM_DOF, frame_bytes);
    pub.send(out, zmq::send_flags::none);
}

} // namespace

int main(int argc, char** argv) {
    const std::string ref_path = (argc > 1) ? argv[1] : DEFAULT_REF_BIN;

    std::size_t num_frames = 0;
    std::vector<float> ref;
    try {
        ref = load_reference(ref_path, num_frames);
    } catch (const std::exception& e) {
        std::cerr << "[ORCH] " << e.what() << std::endl;
        return 1;
    }

    // Optional second positional arg overrides the auto-detected peak frame.
    std::size_t peak_frame = find_peak_extension(ref, num_frames);
    if (argc > 2) {
        peak_frame = static_cast<std::size_t>(std::atoi(argv[2]));
        if (peak_frame >= num_frames) peak_frame = num_frames - 1;
    }

    std::cout << "[ORCH] loaded " << num_frames << " frames x " << NUM_DOF
              << " DoF from " << ref_path << " ("
              << (num_frames / CONTROL_HZ) << " s @ " << CONTROL_HZ << " Hz)"
              << std::endl;
    std::cout << "[ORCH] peak-extension frame = " << peak_frame
              << "  (extension 0->" << peak_frame
              << ", retraction " << peak_frame << "->" << (num_frames - 1) << ")"
              << std::endl;

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    zmq::context_t ctx(1);

    zmq::socket_t sub(ctx, zmq::socket_type::sub);
    sub.connect(TRIGGER_ENDPOINT);
    sub.set(zmq::sockopt::subscribe, "");
    // Short timeout in IDLE so SIGINT is responsive.
    sub.set(zmq::sockopt::rcvtimeo, 200);

    zmq::socket_t pub(ctx, zmq::socket_type::pub);
    pub.bind(POLICY_ENDPOINT);

    std::cout << "[ORCH] SUB " << TRIGGER_ENDPOINT
              << "  (HANDSHAKE / RELEASE)" << std::endl;
    std::cout << "[ORCH] PUB " << POLICY_ENDPOINT
              << "  float32[" << NUM_DOF << "] @ " << CONTROL_HZ << " Hz" << std::endl;

    // Slow-joiner grace period for the policy SUB.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    enum class State { IDLE, EXTENDING, HOLDING, RETRACTING };
    State state = State::IDLE;
    std::size_t cur_frame = 0;
    bool pending_release = false;

    auto handle = [&](TriggerKind kind) {
        if (kind == TriggerKind::Handshake) {
            pending_release = false;
            if (state == State::IDLE) {
                state = State::EXTENDING;
                cur_frame = 0;
                std::cout << "[ORCH] HANDSHAKE -> EXTENDING" << std::endl;
            } else if (state == State::RETRACTING) {
                state = State::HOLDING;
                cur_frame = peak_frame;
                std::cout << "[ORCH] HANDSHAKE (re-engage) -> HOLDING" << std::endl;
            }
        } else if (kind == TriggerKind::Release) {
            if (state == State::HOLDING) {
                state = State::RETRACTING;
                std::cout << "[ORCH] RELEASE -> RETRACTING" << std::endl;
            } else if (state == State::EXTENDING) {
                pending_release = true;
                std::cout << "[ORCH] RELEASE during EXTENDING (deferred)" << std::endl;
            }
        }
    };

    auto next_pub = std::chrono::steady_clock::now();

    while (!g_stop.load(std::memory_order_relaxed)) {
        if (state == State::IDLE) {
            // Block (with rcvtimeo) until a trigger arrives.
            zmq::message_t msg;
            auto r = sub.recv(msg, zmq::recv_flags::none);
            if (r.has_value()) {
                handle(classify_trigger(std::string(
                    static_cast<const char*>(msg.data()), msg.size())));
            }
            if (state == State::IDLE) continue;
            next_pub = std::chrono::steady_clock::now();
        } else {
            // Drain any pending triggers without blocking.
            while (true) {
                zmq::message_t msg;
                auto r = sub.recv(msg, zmq::recv_flags::dontwait);
                if (!r.has_value()) break;
                handle(classify_trigger(std::string(
                    static_cast<const char*>(msg.data()), msg.size())));
            }
            if (state == State::IDLE) continue;
        }

        publish_frame(pub, ref, cur_frame);

        switch (state) {
            case State::EXTENDING:
                if (cur_frame < peak_frame) {
                    ++cur_frame;
                } else {
                    if (pending_release) {
                        state = State::RETRACTING;
                        pending_release = false;
                        std::cout << "[ORCH] EXTENDING done -> RETRACTING (deferred release)" << std::endl;
                    } else {
                        state = State::HOLDING;
                        std::cout << "[ORCH] EXTENDING done -> HOLDING" << std::endl;
                    }
                }
                break;
            case State::HOLDING:
                // cur_frame stays at peak_frame; we keep republishing it.
                break;
            case State::RETRACTING:
                if (cur_frame < num_frames - 1) {
                    ++cur_frame;
                } else {
                    state = State::IDLE;
                    cur_frame = 0;
                    std::cout << "[ORCH] RETRACTING done -> IDLE" << std::endl;
                }
                break;
            case State::IDLE:
                break;
        }

        next_pub += FRAME_PERIOD;
        std::this_thread::sleep_until(next_pub);
    }

    std::cout << "[ORCH] shutting down." << std::endl;
    return 0;
}
