/**
 * PM2 ecosystem config for SAC training.
 *
 * Usage (from repo root, with venv activated in the shell that runs pm2,
 * or set interpreter to your venv's python below):
 *
 *   pm2 start RL/ecosystem.sac.config.cjs
 *   pm2 logs swarm_sac_train
 *   pm2 stop swarm_sac_train
 *
 * Optional: use venv python by setting interpreter to e.g. "miner_env/bin/python"
 * or "/absolute/path/to/venv/bin/python".
 */
const path = require("path");
const projectRoot = path.resolve(__dirname, "..");

module.exports = {
  apps: [
    {
      name: "swarm_sac_train",
      script: path.join(projectRoot, "RL", "train_sac.py"),
      cwd: projectRoot,
      interpreter: "python3",
      args: "--timesteps 500000 --save_path swarm/submission_template/sac_policy.zip",
      autorestart: false,
      watch: false,
      max_restarts: 0,
    },
  ],
};
