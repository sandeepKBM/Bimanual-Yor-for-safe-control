// experiments/baselines/mjpc/cpp/bimanual_cup_task.h
//
// REAL MuJoCo MPC (google-deepmind/mujoco_mpc) task for the YOR bimanual
// cup-collect crossing (SIMULATION ONLY -- no hardware I/O anywhere).
//
// The residual mirrors FL-MPC's cost/constraints as MJPC-style COSTS (MJPC has
// no hard constraints): right-arm joint-reference tracking (the same track_r
// reference FL-MPC gets), left(cup)-arm velocity damping + optional hold,
// cup-arm displacement leash, cup-upright tilt (quadratic + steep violation
// term above the tilt limit), arm-arm clearance over the SAME keypoint spheres
// sim_common.all_pair_barriers uses (+ the held-cup keypoints), table floor,
// torque effort, torque RATE (smoothness; the plant's torque low-pass/slew
// filter is modeled as actuator dynamics, see mjpc_server.cc) and joint accel.
//
// Per-episode / per-step data (reference trajectory, cup keypoints, hold pose,
// cup axis) are passed through mjData.userdata, which MJPC copies into every
// rollout as part of its State. Static keypoint tables live in <custom>
// numerics of the planning model (make_planning_model.py writes both).
#ifndef BIMANUAL_CUP_TASK_H_
#define BIMANUAL_CUP_TASK_H_

#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace bimanual {

// ---- userdata layout (MUST match make_planning_model.py / mjpc_planner.py) ----
inline constexpr int kUdT0 = 0;         // reference start time [s] (== state time at set)
inline constexpr int kUdDtRef = 1;      // reference knot spacing [s]
inline constexpr int kUdNk = 2;         // number of reference knots
inline constexpr int kUdCupAxis = 3;    // 3: cup "up" axis in LEFT EE local frame
inline constexpr int kUdHoldPos = 6;    // 3: leash anchor (left EE world pos at hold)
inline constexpr int kUdQHold = 9;      // 6: left arm hold configuration
inline constexpr int kUdNCup = 15;      // number of cup keypoints
inline constexpr int kUdCupKp = 16;     // kMaxCup x (ox, oy, oz, r) in LEFT EE frame
inline constexpr int kMaxCup = 24;
inline constexpr int kUdRef = kUdCupKp + 4 * kMaxCup;  // nk x (q_r[6], qd_r[6])
inline constexpr int kMaxKnots = 160;
inline constexpr int kNUserdata = kUdRef + 12 * kMaxKnots;

struct Keypoint {
  int body;
  double off[3];
  double r;
};

struct StaticCache {
  std::vector<Keypoint> left, right, left_table, right_table;
  int left_ee = -1;
  int right_ee = -1;
  int lq[6], lv[6], rq[6], rv[6];  // qpos/dof addresses of the 12 arm joints
};

class BimanualCup : public mjpc::Task {
 public:
  explicit BimanualCup(std::string xml_path) : xml_path_(std::move(xml_path)), residual_(this) {}
  std::string Name() const override { return "BimanualCup"; }
  std::string XmlPath() const override { return xml_path_; }

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const BimanualCup* task, std::shared_ptr<const StaticCache> cache = nullptr)
        : mjpc::BaseResidualFn(task), cache_(std::move(cache)) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
    std::shared_ptr<const StaticCache> cache_;
  };

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, cache_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }
  void ResetLocked(const mjModel* model) override;

 private:
  std::string xml_path_;
  std::shared_ptr<const StaticCache> cache_;
  ResidualFn residual_;
};

}  // namespace bimanual

#endif  // BIMANUAL_CUP_TASK_H_
