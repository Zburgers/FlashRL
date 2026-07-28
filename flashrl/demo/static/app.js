const game = document.querySelector("#game");
const gameContext = game.getContext("2d");
const chart = document.querySelector("#score-chart");
const chartContext = chart.getContext("2d");
const actionNames = ["NOOP", "JUMP", "DUCK", "RELEASE"];
let paused = false;
let connected = false;

function text(id, value) {
  document.querySelector(`#${id}`).textContent = value;
}

function drawGame(frame) {
  const width = game.width;
  const height = game.height;
  const ground = height - 64;
  const scaleX = width / 600;
  const scaleY = 2.45;

  gameContext.fillStyle = "#dedbd1";
  gameContext.fillRect(0, 0, width, height);
  gameContext.strokeStyle = "#292925";
  gameContext.lineWidth = 3;
  gameContext.beginPath();
  gameContext.moveTo(0, ground);
  gameContext.lineTo(width, ground);
  gameContext.stroke();

  for (let marker = 0; marker < 14; marker += 1) {
    const x = ((marker * 113 - frame.score * 2.4) % width + width) % width;
    gameContext.fillStyle = marker % 3 === 0 ? "#a8a59c" : "#c3c0b7";
    gameContext.fillRect(x, ground + 18 + (marker % 2) * 10, 14, 3);
  }

  const dino = frame.dino;
  const dinoX = dino.x * scaleX;
  const dinoHeight = dino.height * scaleY;
  const dinoY = ground - dinoHeight - dino.y * scaleY;
  gameContext.fillStyle = "#1b1b19";
  gameContext.fillRect(dinoX, dinoY, dino.width * scaleX, dinoHeight);
  gameContext.fillStyle = "#dedbd1";
  gameContext.fillRect(
    dinoX + dino.width * scaleX - 13,
    dinoY + 13,
    5,
    5,
  );
  gameContext.fillStyle = "#d96743";
  gameContext.fillRect(dinoX + dino.width * scaleX - 8, dinoY + 13, 3, 3);

  frame.obstacles.forEach((obstacle) => {
    const x = obstacle.x * scaleX;
    const obstacleWidth = Math.max(8, obstacle.width * scaleX);
    gameContext.fillStyle = obstacle.type === "bird" ? "#9e4327" : "#30302b";
    if (obstacle.type === "bird") {
      const y = ground - obstacle.y * scaleY - obstacle.height * scaleY;
      gameContext.fillRect(x, y, obstacleWidth, obstacle.height * scaleY);
      gameContext.fillRect(x - 14, y + 12, 18, 7);
      gameContext.fillRect(x + obstacleWidth - 4, y + 12, 18, 7);
    } else {
      gameContext.fillRect(
        x,
        ground - obstacle.height * scaleY,
        obstacleWidth,
        obstacle.height * scaleY,
      );
    }
  });
}

function drawChart(values) {
  chartContext.clearRect(0, 0, chart.width, chart.height);
  if (values.length < 2) return;
  const maximum = Math.max(...values, 1);
  chartContext.strokeStyle = "#d96743";
  chartContext.lineWidth = 3;
  chartContext.beginPath();
  values.forEach((value, index) => {
    const x = (index / (values.length - 1)) * chart.width;
    const y = chart.height - (value / maximum) * (chart.height - 12) - 6;
    if (index === 0) chartContext.moveTo(x, y);
    else chartContext.lineTo(x, y);
  });
  chartContext.stroke();
}

function renderQValues(frame, status) {
  const container = document.querySelector("#q-values");
  if (!frame.q_values) {
    container.innerHTML =
      '<p class="empty">This policy is deterministic code. Load a DQN checkpoint to inspect learned action values.</p>';
    text("q-mode", "Heuristic policy");
    return;
  }
  text("q-mode", "Network output");
  const minimum = Math.min(...frame.q_values);
  const shifted = frame.q_values.map((value) => value - minimum + 0.01);
  const maximum = Math.max(...shifted);
  container.innerHTML = frame.q_values
    .map((value, index) => {
      const selected = index === frame.action.id ? " selected" : "";
      const width = Math.max(2, (shifted[index] / maximum) * 100);
      const name = status.action_names[index] || actionNames[index];
      return `<div class="q-row${selected}">
        <span>${name}</span>
        <span class="q-bar" style="width:${width}%"></span>
        <span class="q-number">${value.toFixed(3)}</span>
      </div>`;
    })
    .join("");
}

function renderIdentity(policy) {
  const container = document.querySelector("#identity-grid");
  const entries = Object.entries(policy);
  container.innerHTML = entries
    .map(
      ([key, value]) => `<div class="identity-item">
        <span>${key.replaceAll("_", " ")}</span>
        <strong>${String(value)}</strong>
      </div>`,
    )
    .join("");
}

function render(frame, status) {
  drawGame(frame);
  drawChart(frame.score_history);
  renderQValues(frame, status);
  renderIdentity(frame.policy);
  text("score", frame.score.toFixed(1));
  text("seed", frame.seed);
  text("episode", frame.episode);
  text("step", frame.step);
  text("action", frame.action.name);
  text("reward", `Reward ${frame.reward >= 0 ? "+" : ""}${frame.reward.toFixed(3)}`);
  text("speed-metric", frame.speed.toFixed(2));
  text("survival", `${frame.survival_time_s.toFixed(1)}s`);
  text("jumping", frame.dino.jumping ? "Yes" : "No");
  text("ducking", frame.dino.ducking ? "Yes" : "No");
  text("policy-name", `${status.policy.name} policy`);
  text("live-state", status.paused ? "Paused" : "Live");
  document.querySelector("#pause").textContent = status.paused ? "Resume" : "Pause";
  document.querySelector("#seed-input").value = status.seed;
  document.querySelector("#canvas-message").hidden = !status.paused;
  document.querySelector("#canvas-message").textContent = status.paused
    ? "Simulation paused"
    : "";
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed: ${response.status}`);
  return payload;
}

async function update() {
  try {
    const [frame, status] = await Promise.all([
      request("/api/frame"),
      request("/api/status"),
    ]);
    render(frame, status);
    paused = status.paused;
    connected = true;
    document.querySelector("#error-banner").hidden = true;
  } catch (error) {
    connected = false;
    document.querySelector("#error-banner").hidden = false;
    text("live-state", "Offline");
    console.error(error);
  } finally {
    window.setTimeout(update, connected ? 80 : 1000);
  }
}

document.querySelector("#controls-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const seed = Number.parseInt(document.querySelector("#seed-input").value, 10);
  try {
    await request("/api/reset", {
      method: "POST",
      body: JSON.stringify({ seed }),
    });
  } catch (error) {
    window.alert(error.message);
  }
});

document.querySelector("#speed-input").addEventListener("change", async (event) => {
  try {
    await request("/api/control", {
      method: "POST",
      body: JSON.stringify({ speed: Number.parseFloat(event.target.value) }),
    });
  } catch (error) {
    window.alert(error.message);
  }
});

document.querySelector("#pause").addEventListener("click", async () => {
  try {
    await request("/api/control", {
      method: "POST",
      body: JSON.stringify({ paused: !paused }),
    });
  } catch (error) {
    window.alert(error.message);
  }
});

update();
