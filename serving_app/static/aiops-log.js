// logs/aiops.log 파서 (#56 운영 현황 탭). 줄 형식: "%(asctime)s [%(levelname)s] [TAG] message"
// 태그·문구는 monitoring/retrain_trigger.py가 소유한다. 형식이 바뀌면 tests/test_dashboard_log_parser.py가 깨진다.
(function (root) {
  const LINE = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[\w+\] \[([A-Z_]+)\] (.*)$/;
  const DRIFT = /week1_rmse=([\d.]+) > naive_rmse=([\d.]+).*period=([^,]+), model=([^)]+)\)/;
  const LEVEL = { WARN: "warn", GATE_FAILED: "warn", ERROR: "err", OK: "ok", INFO: "info" };
  // [ALERT]는 같은 결과([OK]/[GATE_FAILED]/[ERROR])를 운영자 채널로 보낸 사본, data selected/excluded는 세부 기록
  const SKIP = (tag, msg) => tag === "ALERT" || /^retrain data (selected|excluded)/.test(msg);

  function parseAiopsLog(text) {
    const events = [];
    const drift = [];
    for (const line of String(text).split("\n")) {
      const m = LINE.exec(line);
      if (!m) continue; // Traceback 등 이어지는 줄
      const [, time, tag, message] = m;
      if (!(tag in LEVEL) || SKIP(tag, message)) continue;
      events.push({ time, tag, level: LEVEL[tag], message });
      const d = tag === "WARN" && DRIFT.exec(message);
      if (d) drift.push({ time, week1: Number(d[1]), naive: Number(d[2]), period: d[3], model: d[4] });
    }
    return { events: events.reverse(), drift };
  }

  if (typeof module !== "undefined") module.exports = { parseAiopsLog };
  else root.parseAiopsLog = parseAiopsLog;
})(this);
