// experiments/baselines/mjpc/cpp/bimanual_cup_task.cc -- see the header.
#include "bimanual_cup_task.h"

#include <algorithm>
#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace bimanual {

namespace {

// read a flat (bid, ox, oy, oz, r) x N custom numeric into a keypoint list
std::vector<Keypoint> ReadKps(const mjModel* m, const char* name) {
  std::vector<Keypoint> out;
  int id = mj_name2id(m, mjOBJ_NUMERIC, name);
  if (id < 0) mju_error("BimanualCup: missing custom numeric '%s'", name);
  int n = m->numeric_size[id];
  const double* d = m->numeric_data + m->numeric_adr[id];
  if (n % 5 != 0) mju_error("BimanualCup: numeric '%s' size not a multiple of 5", name);
  for (int i = 0; i < n / 5; i++) {
    Keypoint k;
    k.body = static_cast<int>(d[5 * i]);
    k.off[0] = d[5 * i + 1];
    k.off[1] = d[5 * i + 2];
    k.off[2] = d[5 * i + 3];
    k.r = d[5 * i + 4];
    out.push_back(k);
  }
  return out;
}

inline void KpWorld(const mjData* d, int body, const double off[3], double out[3]) {
  mju_mulMatVec3(out, d->xmat + 9 * body, off);
  mju_addTo3(out, d->xpos + 3 * body);
}

inline double Relu(double x) { return x > 0 ? x : 0; }

// linear interpolation of the (q, qd) reference knots stored in userdata
void RefAt(const mjData* d, double t, double q[6], double qd[6]) {
  const double* ud = d->userdata;
  double t0 = ud[kUdT0], dt = ud[kUdDtRef];
  int nk = static_cast<int>(ud[kUdNk]);
  if (nk <= 0) {
    for (int i = 0; i < 6; i++) q[i] = qd[i] = 0;
    return;
  }
  double s = (t - t0) / std::max(dt, 1e-9);
  if (s <= 0) s = 0;
  if (s >= nk - 1) s = nk - 1;
  int i0 = static_cast<int>(std::floor(s));
  if (i0 >= nk - 1) i0 = nk - 1;
  int i1 = std::min(i0 + 1, nk - 1);
  double a = s - i0;
  const double* k0 = ud + kUdRef + 12 * i0;
  const double* k1 = ud + kUdRef + 12 * i1;
  for (int i = 0; i < 6; i++) {
    q[i] = (1 - a) * k0[i] + a * k1[i];
    qd[i] = (1 - a) * k0[6 + i] + a * k1[6 + i];
  }
}

}  // namespace

void BimanualCup::ResetLocked(const mjModel* model) {
  auto c = std::make_shared<StaticCache>();
  c->left = ReadKps(model, "kp_left");
  c->right = ReadKps(model, "kp_right");
  c->left_table = ReadKps(model, "kp_left_table");
  c->right_table = ReadKps(model, "kp_right_table");
  c->left_ee = mj_name2id(model, mjOBJ_BODY, "left_arm_gripper_base");
  c->right_ee = mj_name2id(model, mjOBJ_BODY, "right_arm_gripper_base");
  for (int i = 0; i < 6; i++) {
    std::string ln = "left_arm_joint" + std::to_string(i + 1);
    std::string rn = "right_arm_joint" + std::to_string(i + 1);
    int lj = mj_name2id(model, mjOBJ_JOINT, ln.c_str());
    int rj = mj_name2id(model, mjOBJ_JOINT, rn.c_str());
    if (lj < 0 || rj < 0) mju_error("BimanualCup: arm joint missing");
    c->lq[i] = model->jnt_qposadr[lj];
    c->lv[i] = model->jnt_dofadr[lj];
    c->rq[i] = model->jnt_qposadr[rj];
    c->rv[i] = model->jnt_dofadr[rj];
  }
  if (model->nuserdata < kNUserdata) mju_error("BimanualCup: nuserdata too small");
  cache_ = c;
  residual_.cache_ = c;
}

void BimanualCup::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                       double* residual) const {
  const StaticCache& c = *cache_;
  const double* ud = data->userdata;
  // parameters (residual_* numerics, in XML order)
  const double h_safe = parameters_[0];     // clearance activation margin [m] (on h, which already includes D_MIN)
  const double d_min = parameters_[1];      // D_MIN (sim_common) [m]
  const double tilt_lim = parameters_[2];   // tilt limit [rad] for the steep violation term
  const double leash = parameters_[3];      // per-axis leash [m]
  const double z_floor = parameters_[4];    // table z threshold (table_top - margin)

  int counter = 0;

  // ---- 0/1: right-arm joint tracking of the task reference (track_r) ----
  double qref[6], qdref[6];
  RefAt(data, data->time, qref, qdref);
  for (int i = 0; i < 6; i++) residual[counter + i] = data->qpos[c.rq[i]] - qref[i];
  counter += 6;
  for (int i = 0; i < 6; i++) residual[counter + i] = data->qvel[c.rv[i]] - qdref[i];
  counter += 6;

  // ---- 2: left (cup) arm velocity damping ----
  for (int i = 0; i < 6; i++) residual[counter + i] = data->qvel[c.lv[i]];
  counter += 6;

  // ---- 3: left arm hold (weight 0 by default == FL-MPC's ego_hold=False) ----
  for (int i = 0; i < 6; i++) residual[counter + i] = data->qpos[c.lq[i]] - ud[kUdQHold + i];
  counter += 6;

  // ---- 4: cup-arm displacement leash (per world axis, rectified) ----
  const double* pee = data->xpos + 3 * c.left_ee;
  for (int i = 0; i < 3; i++) {
    double dp = pee[i] - ud[kUdHoldPos + i];
    residual[counter + i] = Relu(std::fabs(dp) - leash);
  }
  counter += 3;

  // ---- 5/6: cup upright: tilt vector (-a_y, a_x) of the cup axis in world ----
  double a_world[3];
  mju_mulMatVec3(a_world, data->xmat + 9 * c.left_ee, ud + kUdCupAxis);
  double tx = -a_world[1], ty = a_world[0];
  residual[counter + 0] = tx;
  residual[counter + 1] = ty;
  counter += 2;
  double tilt = std::atan2(std::sqrt(tx * tx + ty * ty), a_world[2]);
  residual[counter++] = Relu(tilt - tilt_lim);

  // ---- 7: arm-arm clearance over the keypoint-sphere proxy (+ cup kps) ----
  // h = dist - (r_e + r_i + D_MIN)  (identical to sim_common.all_pair_barriers)
  // residual = sqrt(sum relu(h_safe - h)^2)  -> quadratic norm = 0.5*sum relu^2
  {
    double pr[64][3];
    int nr = std::min<int>(c.right.size(), 64);
    for (int j = 0; j < nr; j++) KpWorld(data, c.right[j].body, c.right[j].off, pr[j]);
    double acc = 0.0;
    auto accum = [&](const double pe[3], double re) {
      for (int j = 0; j < nr; j++) {
        double dx = pe[0] - pr[j][0], dy = pe[1] - pr[j][1], dz = pe[2] - pr[j][2];
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        double h = dist - (re + c.right[j].r + d_min);
        double v = Relu(h_safe - h);
        acc += v * v;
      }
    };
    double pe[3];
    for (const Keypoint& k : c.left) {
      KpWorld(data, k.body, k.off, pe);
      accum(pe, k.r);
    }
    int ncup = std::min(static_cast<int>(ud[kUdNCup]), kMaxCup);
    for (int i = 0; i < ncup; i++) {
      const double* ck = ud + kUdCupKp + 4 * i;
      KpWorld(data, c.left_ee, ck, pe);
      accum(pe, ck[3]);
    }
    residual[counter++] = std::sqrt(acc);
  }

  // ---- 8: table floor (both arms' keypoints above z_floor) ----
  {
    double acc = 0.0, p[3];
    for (const Keypoint& k : c.left_table) {
      KpWorld(data, k.body, k.off, p);
      double v = Relu(z_floor - p[2]);
      acc += v * v;
    }
    for (const Keypoint& k : c.right_table) {
      KpWorld(data, k.body, k.off, p);
      double v = Relu(z_floor - p[2]);
      acc += v * v;
    }
    residual[counter++] = std::sqrt(acc);
  }

  // ---- 9: torque effort on the APPLIED (filtered) joint torque, Nm ----
  for (int i = 0; i < model->na; i++) residual[counter + i] = data->act[i];
  counter += model->na;

  // ---- 10: torque RATE of the applied (filtered) torque, Nm/s ----
  for (int i = 0; i < model->na; i++) residual[counter + i] = data->act_dot[i];
  counter += model->na;

  // ---- 11: joint accelerations of the 12 arm joints ----
  for (int i = 0; i < 6; i++) residual[counter + i] = data->qacc[c.lv[i]];
  for (int i = 0; i < 6; i++) residual[counter + 6 + i] = data->qacc[c.rv[i]];
  counter += 12;

  // sensor-dimension sanity check
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) user_sensor_dim += model->sensor_dim[i];
  }
  if (user_sensor_dim != counter) {
    mju_error_i("BimanualCup: user-sensor dim mismatch, residual length %d", counter);
  }
}

}  // namespace bimanual
