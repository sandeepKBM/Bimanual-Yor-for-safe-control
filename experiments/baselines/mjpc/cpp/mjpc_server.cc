// experiments/baselines/mjpc/cpp/mjpc_server.cc
//
// Headless, synchronous MuJoCo MPC (MJPC) planner server for the bimanual
// cup-collect baseline. SIMULATION ONLY: it reads a planning-model XML and
// talks binary over stdin/stdout to the Python driver
// (experiments/baselines/mjpc/mjpc_planner.py). No sockets, no hardware.
//
// Pattern follows mjpc/testspeed.cc (MJPC's own headless driver): Agent with a
// task list, mjcb_sensor -> task residual, agent.PlanIteration(&pool),
// ActivePlanner().ActionFromPolicy(...).
//
// Plant torque filter as actuator dynamics: the demo applies every commanded
// torque through sim_common.TorqueCommandFilter (first-order low-pass
// alpha=0.25 @ 5 ms + 400 Nm/s slew limit). The planning model's 12 arm
// motors use dyntype="user" + actearly; mjcb_act_dyn below implements that
// exact filter (Euler-integrated act == filter state y), so MJPC plans through
// the same torque smoothing the plant really applies. act_dot is then the
// applied-torque RATE, penalised by the TauRate residual (control-rate term).
//
// Protocol (little-endian, native):
//   startup  -> server writes int32[6] {nq, nv, na, nu, nuserdata, nmocap}
//   cmd 1 STEP : int32 n_iter, double[1+nq+nv+na+nuserdata+7*nmocap]
//                {time,qpos,qvel,act,userdata,mocap_pos(3*nmocap),mocap_quat(4*nmocap)}
//                -> int32 status, double[nu] ctrl, double plan_ms, double best_cost
//   cmd 2 SETW : int32 len, char[len] name, double value -> int32 status (weight)
//   cmd 3 RESET: -> int32 status (nominal policy reset to zero torque)
//   cmd 4 SETP : int32 len, char[len] name, double value -> int32 status (residual_ param)
//   cmd 5 DEBUG: weighted cost terms of the best trajectory (see below)
//   cmd 6 RESET_TO: double[nu] initial repeated action -> int32 status
//   cmd 0 EXIT
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

#include "bimanual_cup_task.h"

namespace {

mjpc::Task* g_task = nullptr;

void ResidualCallback(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) g_task->Residual(model, data, data->sensordata);
}

// plant filter constants (sim_common.FILTER_LOWPASS_ALPHA / FILTER_RATE_LIMIT_NM_S, DT)
constexpr double kPlantDt = 0.005;
constexpr double kAlpha = 0.25;
constexpr double kRate = 400.0;

// actuator parameterisation, read once from the planning model's <custom>:
//   bimanual_act_mode = 0 -> ctrl IS the joint torque [Nm]
//   bimanual_act_mode = 1 -> ctrl is a joint SETPOINT; torque = clip(kp(u-q) - kd*qd, +-100)
//                           (MJPC's usual position-actuator parameterisation; the plant
//                           computes the identical PD torque from the returned setpoint)
// Either way the resulting torque then passes through the plant's filter.
int g_act_mode = 0;
std::vector<double> g_kp, g_kd;

void ReadActParams(const mjModel* m) {
  g_act_mode = (int)mjpc::GetNumberOrDefault(0, m, "bimanual_act_mode");
  g_kp.assign(m->nu, 0.0);
  g_kd.assign(m->nu, 0.0);
  int ikp = mj_name2id(m, mjOBJ_NUMERIC, "bimanual_kp");
  int ikd = mj_name2id(m, mjOBJ_NUMERIC, "bimanual_kd");
  for (int i = 0; i < m->nu; i++) {
    if (ikp >= 0 && i < m->numeric_size[ikp]) g_kp[i] = m->numeric_data[m->numeric_adr[ikp] + i];
    if (ikd >= 0 && i < m->numeric_size[ikd]) g_kd[i] = m->numeric_data[m->numeric_adr[ikd] + i];
  }
}

mjtNum FilterActDyn(const mjModel* m, const mjData* d, int id) {
  int adr = m->actuator_actadr[id];
  double y = d->act[adr];
  double u = d->ctrl[id];
  if (m->actuator_ctrllimited[id]) {
    u = mju_clip(u, m->actuator_ctrlrange[2 * id], m->actuator_ctrlrange[2 * id + 1]);
  }
  if (g_act_mode == 1) {
    int j = m->actuator_trnid[2 * id];
    double q = d->qpos[m->jnt_qposadr[j]], qd = d->qvel[m->jnt_dofadr[j]];
    u = mju_clip(g_kp[id] * (u - q) - g_kd[id] * qd, -100.0, 100.0);
  }
  double dt = m->opt.timestep;
  double a_eff = 1.0 - std::pow(1.0 - kAlpha, dt / kPlantDt);
  double dx = a_eff * (u - y);
  double dmax = kRate * dt;
  dx = mju_clip(dx, -dmax, dmax);
  return dx / dt;
}

bool ReadAll(void* buf, size_t n) { return std::fread(buf, 1, n, stdin) == n; }
void WriteAll(const void* buf, size_t n) { std::fwrite(buf, 1, n, stdout); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: mjpc_bimanual_server <planning_model.xml> [threads]\n");
    return 2;
  }
  std::string xml = argv[1];
  int nthreads = argc > 2 ? std::atoi(argv[2]) : 8;

  auto task = std::make_shared<bimanual::BimanualCup>(xml);
  mjpc::Agent agent;
  agent.SetTaskList({task});
  agent.gui_task_id = 0;
  mjpc::Agent::LoadModelResult load = agent.LoadModel();
  mjModel* model = load.model.get();
  if (!model) {
    std::fprintf(stderr, "load error: %s\n", load.error.c_str());
    return 3;
  }
  mjData* data = mj_makeData(model);
  mj_forward(model, data);

  ReadActParams(model);
  mjcb_act_dyn = &FilterActDyn;
  agent.estimator_enabled = false;
  agent.Initialize(model);
  agent.Allocate();
  std::vector<double> zeros(model->nu, 0.0);
  agent.Reset(zeros.data());
  agent.plan_enabled = true;
  agent.action_enabled = true;
  g_task = agent.ActiveTask();
  mjcb_sensor = &ResidualCallback;

  mjpc::ThreadPool pool(nthreads);
  std::fprintf(stderr, "[mjpc_server] mujoco %s, planner %d, threads %d, nq %d nv %d na %d nu %d nuserdata %d, horizon %.3f, dt %.4f\n",
               mj_versionString(), (int)mjpc::GetNumberOrDefault(0, model, "agent_planner"), nthreads,
               model->nq, model->nv, model->na, model->nu, (int)model->nuserdata,
               agent.Horizon(), mjpc::GetNumberOrDefault(0.01, model, "agent_timestep"));

  int dims[6] = {model->nq, model->nv, model->na, model->nu, (int)model->nuserdata, model->nmocap};
  WriteAll(dims, sizeof(dims));
  std::fflush(stdout);

  int nstate = 1 + model->nq + model->nv + model->na + model->nuserdata + 7 * model->nmocap;
  std::vector<double> buf(nstate);
  std::vector<double> ctrl(model->nu);

  while (true) {
    int cmd;
    if (!ReadAll(&cmd, sizeof(int))) break;
    if (cmd == 0) break;
    if (cmd == 1) {
      int n_iter;
      if (!ReadAll(&n_iter, sizeof(int))) break;
      if (!ReadAll(buf.data(), sizeof(double) * nstate)) break;
      const double* p = buf.data();
      data->time = p[0];
      p += 1;
      mju_copy(data->qpos, p, model->nq);
      p += model->nq;
      mju_copy(data->qvel, p, model->nv);
      p += model->nv;
      mju_copy(data->act, p, model->na);
      p += model->na;
      mju_copy(data->userdata, p, model->nuserdata);
      p += model->nuserdata;
      mju_copy(data->mocap_pos, p, 3 * model->nmocap);
      p += 3 * model->nmocap;
      mju_copy(data->mocap_quat, p, 4 * model->nmocap);
      agent.ActiveTask()->Transition(model, data);
      agent.state.Set(model, data);
      auto t0 = std::chrono::steady_clock::now();
      for (int k = 0; k < n_iter; k++) agent.PlanIteration(&pool);
      double plan_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t0).count();
      agent.ActivePlanner().ActionFromPolicy(ctrl.data(), agent.state.state().data(),
                                             agent.state.time(), /*use_previous=*/false);
      const mjpc::Trajectory* best = agent.ActivePlanner().BestTrajectory();
      double best_cost = best ? best->total_return : NAN;
      int status = 0;
      WriteAll(&status, sizeof(int));
      WriteAll(ctrl.data(), sizeof(double) * model->nu);
      WriteAll(&plan_ms, sizeof(double));
      WriteAll(&best_cost, sizeof(double));
      std::fflush(stdout);
    } else if (cmd == 2 || cmd == 4) {
      int len;
      if (!ReadAll(&len, sizeof(int))) break;
      std::string name(len, '\0');
      if (!ReadAll(name.data(), len)) break;
      double v;
      if (!ReadAll(&v, sizeof(double))) break;
      int status = (cmd == 2) ? agent.SetWeightByName(name, v) : agent.SetParamByName(name, v);
      agent.ActiveTask()->UpdateResidual();
      WriteAll(&status, sizeof(int));
      std::fflush(stdout);
    } else if (cmd == 5) {
      // DEBUG: weighted cost terms of the best trajectory, summed over its horizon
      // (+ the first action and a few first-step residuals). -> int32 nterm, double[nterm], int32 nact, double[nact]
      mjpc::Task* task = agent.ActiveTask();
      const mjpc::Trajectory* best = agent.ActivePlanner().BestTrajectory();
      int nterm = task->num_term;
      std::vector<double> sum(nterm, 0.0), terms(nterm);
      if (best) {
        for (int t = 0; t < best->horizon; t++) {
          task->CostTerms(terms.data(), best->residual.data() + t * task->num_residual);
          for (int i = 0; i < nterm; i++) sum[i] += terms[i];
        }
      }
      WriteAll(&nterm, sizeof(int));
      WriteAll(sum.data(), sizeof(double) * nterm);
      int nact = model->nu;
      std::vector<double> a0(nact, 0.0);
      if (best && best->horizon > 1) for (int i = 0; i < nact; i++) a0[i] = best->actions[i];
      WriteAll(&nact, sizeof(int));
      WriteAll(a0.data(), sizeof(double) * nact);
      std::fflush(stdout);
    } else if (cmd == 7) {
      // DEBUG: rank candidates for the current state -> int32 n, double[n] sorted scores
      auto* rp = dynamic_cast<mjpc::RankedPlanner*>(&agent.ActivePlanner());
      int n = 0;
      std::vector<double> sc(16, NAN);
      if (rp) {
        agent.ActivePlanner().SetState(agent.state);
        n = rp->OptimizePolicyCandidates(16, agent.PlanSteps(), pool);
        for (int i = 0; i < n; i++) sc[i] = rp->CandidateScore(i);
      }
      WriteAll(&n, sizeof(int));
      WriteAll(sc.data(), sizeof(double) * n);
      std::fflush(stdout);
    } else if (cmd == 6) {
      std::vector<double> a0(model->nu);
      if (!ReadAll(a0.data(), sizeof(double) * model->nu)) break;
      agent.Reset(a0.data());
      int status = 0;
      WriteAll(&status, sizeof(int));
      std::fflush(stdout);
    } else if (cmd == 3) {
      agent.Reset(zeros.data());
      int status = 0;
      WriteAll(&status, sizeof(int));
      std::fflush(stdout);
    } else {
      std::fprintf(stderr, "[mjpc_server] unknown cmd %d\n", cmd);
      break;
    }
  }
  mjcb_sensor = nullptr;
  mjcb_act_dyn = nullptr;
  mj_deleteData(data);
  return 0;
}
