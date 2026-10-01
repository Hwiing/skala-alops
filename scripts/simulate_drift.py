"""
AIOps 2/3 (#16): 4종 드리프트 시나리오 + 정상 대조군을 서빙 서버에 보내 판정·재학습 반응을 기록한다.

시나리오 (모두 합성 주입, 고정 seed - 실측 성능 증빙이 아님)
    normal             정상 대조군: 작은 랜덤워크
    oil_shock          국제유가 급등: 싱가포르 경유 +35% (3일에 걸쳐)
    fx_shock           환율 급변: 원/달러 +10%
    tax_policy         유류세 정책 변경: 경유 탄력세율 인하율 10% → 0% (인하 종료)
    supply_disruption  공급 차질: 국제가·환율 변화 없이 소매가 +120원, 2주에 걸쳐 절반 회복

소매가(diesel_price)는 원가(싱가포르가 × 환율 / 158.987 L)와 세금 변화를 매일 일부만 따라가는
부분조정 모형으로 만든다. 실제 주유소 가격이 국제가를 1~2주 늦게 따라가는 현상을 흉내 낸 것.

주의: 입력이 급변했다고 해서 드리프트는 아니다. 드리프트 판정은 서버가 1주차 RMSE와 같은
기간 naive RMSE를 비교해서 내린다. naive도 같이 틀리는 급변이면 drift가 아닐 수 있다.

배치 길이 175행 = 입력 120 + 짝 28 − 1 + 4주 정답 28 (contracts.md v2, #34 BATCH_MIN_ROWS).
충격일(SHOCK_DAY)은 짝의 기준일(120~147번째 행) 사이에 둬서, 앞쪽 짝들의 정답에 충격이 들어가게 한다.

사전 준비: 서빙 서버 실행 (MODEL_SOURCE=mlflow 권장)
실행 예:
    python scripts/simulate_drift.py --dry-run                     # 서버 없이 시나리오 입력만 확인
    python scripts/simulate_drift.py --url http://localhost:8001/predict/batch-test
    python scripts/simulate_drift.py --scenario oil_shock --out logs/scenario_results.md
    python scripts/simulate_drift.py --base-csv data/processed/diesel_features_2008_spliced.csv
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import urllib.error
import urllib.request
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.contracts import BATCH_MIN_ROWS  # noqa: E402

API_URL = "http://localhost:8000/predict/batch-test"
BATCH_N = BATCH_MIN_ROWS  # 175
SHOCK_DAY = 125
L_PER_BBL = 158.987
FULL_TAX_WON = 528.75  # 경유 유류세(인하 전, 원/L) - 인하율 1%p ≈ 5.3원/L
PASS_THROUGH = 0.15  # 소매가가 하루에 목표가(원가+세금 변화) 차이의 15%씩 따라감

SCENARIOS = {
    "normal": "정상 대조군 (작은 랜덤워크)",
    "oil_shock": "국제유가 급등: 싱가포르 경유 +35% (3일)",
    "fx_shock": "환율 급변: 원/달러 +10%",
    "tax_policy": "유류세 정책: 탄력세율 인하율 10% → 0%",
    "supply_disruption": "공급 차질: 소매가 +120원, 2주 반감",
}

DEFAULT_BASE = {
    "diesel_price": 1550.0,
    "singapore_diesel_price": 90.0,
    "usd_krw": 1400.0,
    "tax_or_supply_feature": 10.0,
}
DEFAULT_SIGMA = {"singapore_diesel_price": 0.012, "usd_krw": 0.003, "diesel_price": 1.5}


def base_from_csv(path: str) -> tuple[dict, dict]:
    """실측 CSV의 마지막 행을 출발점으로, 최근 1년 일별 변동성을 정상 변동성으로 쓴다."""
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))[-366:]
    last = rows[-1]
    base = {k: float(last[k]) for k in DEFAULT_BASE}

    def log_sigma(key):
        vals = [float(r[key]) for r in rows]
        rets = [math.log(b / a) for a, b in zip(vals, vals[1:]) if a > 0 and b > 0]
        mean = sum(rets) / len(rets)
        return math.sqrt(sum((r - mean) ** 2 for r in rets) / len(rets))

    sigma = {
        "singapore_diesel_price": log_sigma("singapore_diesel_price"),
        "usd_krw": log_sigma("usd_krw"),
        "diesel_price": DEFAULT_SIGMA["diesel_price"],
    }
    return base, sigma


def _cost(sg: float, fx: float) -> float:
    return sg * fx / L_PER_BBL


def generate_rows(
    scenario: str,
    n: int = BATCH_N,
    seed: int = 42,
    start: str = "2026-01-01",
    base: dict | None = None,
    sigma: dict | None = None,
    shock_day: int = SHOCK_DAY,
) -> list[dict]:
    """시나리오별 date + 4피처 행 n개 (하루 간격). 같은 seed면 항상 같은 값."""
    if scenario not in SCENARIOS:
        raise ValueError(f"알 수 없는 시나리오: {scenario} (가능: {', '.join(SCENARIOS)})")
    base = base or DEFAULT_BASE
    sigma = sigma or DEFAULT_SIGMA
    rng = random.Random(seed)  # 시나리오끼리 같은 잡음을 써야 충격만의 효과를 비교할 수 있다
    first = date.fromisoformat(start)

    sg, fx, cut = base["singapore_diesel_price"], base["usd_krw"], base["tax_or_supply_feature"]
    retail = base["diesel_price"]
    anchor = retail - _cost(sg, fx)  # 목표가 = 원가 + anchor (+ 세금·공급 충격)
    supply = 0.0
    rows = []
    for i in range(n):
        sg *= math.exp(rng.gauss(0, sigma["singapore_diesel_price"]))
        fx *= math.exp(rng.gauss(0, sigma["usd_krw"]))
        noise = rng.gauss(0, sigma["diesel_price"])

        if scenario == "oil_shock" and shock_day <= i < shock_day + 3:
            sg *= 1.35 ** (1 / 3)
        if scenario == "fx_shock" and i == shock_day:
            fx *= 1.10
        if scenario == "tax_policy" and i == shock_day:
            cut = 0.0
        if scenario == "supply_disruption":
            if i == shock_day:
                supply = 120.0
            elif i > shock_day:
                supply *= 0.5 ** (1 / 14)

        tax_up = (base["tax_or_supply_feature"] - cut) / 100 * FULL_TAX_WON
        target = _cost(sg, fx) + anchor + tax_up
        retail += PASS_THROUGH * (target - retail) + noise
        if scenario == "tax_policy" and i == shock_day:
            retail += tax_up * 0.5  # 세금 변화는 원가보다 빨리(당일 절반) 반영된다
        shown = retail + supply

        rows.append(
            {
                "date": (first + timedelta(days=i)).isoformat(),
                "diesel_price": round(shown, 2),
                "singapore_diesel_price": round(sg, 2),
                "usd_krw": round(fx, 2),
                "tax_or_supply_feature": cut,
            }
        )
    return rows


def replay_rows(path: str, end: str, n: int = BATCH_N) -> list[dict]:
    """실측 CSV에서 end일까지 n행을 그대로 잘라 배치로 쓴다 (합성 충격 없음).

    기준일은 앞에서 120번째 행부터 28개 → end − 55일 ~ end − 28일. 실제 그 시기에 서빙했다면
    받았을 판정을 재현한다. 서빙 모델이 그 시기 이후 데이터로 학습됐다면 결과가 낙관적이다.
    """
    with open(path, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["date"] <= end]
    if len(rows) < n or rows[-1]["date"] != end:
        raise ValueError(
            f"{end}까지 {n}행이 필요합니다 (마지막 {rows[-1]['date'] if rows else '-'})"
        )
    keys = ("diesel_price", "singapore_diesel_price", "usd_krw", "tax_or_supply_feature")
    return [{"date": r["date"], **{k: float(r[k]) for k in keys}} for r in rows[-n:]]


def describe(rows: list[dict], control: list[dict]) -> str:
    """같은 seed의 정상 대조군 대비 충격 효과 (충격일 이후 가장 크게 벌어진 값).

    랜덤워크 자체의 흐름을 빼야 '충격 때문에 바뀐 양'만 보인다.
    """
    if rows == control:
        return "충격 없음 (대조군)"

    def max_gap(key, pct):
        gaps = [
            (r[key] / c[key] - 1) * 100 if pct else r[key] - c[key]
            for r, c in zip(rows[SHOCK_DAY - 1 :], control[SHOCK_DAY - 1 :])
        ]
        return max(gaps, key=abs)

    return (
        f"소매 {max_gap('diesel_price', False):+.0f}원, "
        f"싱가포르 {max_gap('singapore_diesel_price', True):+.1f}%, "
        f"환율 {max_gap('usd_krw', True):+.1f}%, "
        f"인하율 {control[-1]['tax_or_supply_feature']:g}→{rows[-1]['tax_or_supply_feature']:g}%"
    )


def send_batch(rows: list[dict], label: str, url: str = API_URL, timeout: float = 180) -> dict:
    """rows를 batch-test로 POST. 성공·HTTP 오류·연결 실패를 모두 dict로 돌려준다 (예외 없음).

    timeout이 긴 이유: 드리프트로 재학습(fine-tuning)이 걸리면 응답까지 수십 초 걸린다.
    """
    body = json.dumps({"rows": rows}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "label": label,
                "http_status": resp.status,
                "drift_check": data.get("drift_check", {}),
            }
    except urllib.error.HTTPError as exc:  # 422 입력 오류, 501 미구현, 503 모델 없음 등
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        return {"label": label, "http_status": exc.code, "error": detail}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"label": label, "http_status": None, "error": f"연결 실패: {exc}"}


def summarize(result: dict, inputs: str) -> dict:
    """응답 DriftCheck(공개 계약)를 결과표 한 줄로. 판정은 상태에서 거꾸로 읽는다."""
    check = result.get("drift_check") or {}
    status = check.get("status")
    if status == "ok":
        detection = "ok"
    elif status == "insufficient_data" and check.get("drift_rmse") is None:
        detection = "보류"
    elif status:
        detection = "drift"
    else:
        detection = "-"
    reload = check.get("reload") or {}
    return {
        "scenario": result["label"],
        "inputs": inputs,
        "http": result.get("http_status"),
        "detection": detection,
        "week1_rmse": check.get("drift_rmse"),
        "naive_rmse": check.get("drift_naive_rmse"),
        "action": status or result.get("error", "-"),
        "version": check.get("version") or "-",
        "reload": reload.get("version") if reload.get("reloaded") else ("실패" if reload else "-"),
    }


def to_markdown(
    summaries: list[dict], meta: str, input_label: str = "입력 충격(대조군 대비 최대)"
) -> str:
    def f(v):
        return f"{v:.2f}" if isinstance(v, float) else ("-" if v is None else str(v))

    head = f"| 시나리오 | {input_label} | HTTP | 판정 | 1주차 RMSE | naive RMSE | 조치 | 승격 | 서빙 교체 |"
    lines = [f"> {meta}", "", head, "|" + "---|" * 9]
    for s in summaries:
        lines.append(
            "| "
            + " | ".join(
                f(s[k])
                for k in [
                    "scenario",
                    "inputs",
                    "http",
                    "detection",
                    "week1_rmse",
                    "naive_rmse",
                    "action",
                    "version",
                    "reload",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default=API_URL)
    ap.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="2026-01-01", help="첫 시나리오 배치의 시작일")
    ap.add_argument("--base-csv", help="실측 CSV 마지막 행·최근 1년 변동성을 출발점으로 사용")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--dry-run", action="store_true", help="서버에 보내지 않고 입력 요약만 출력")
    ap.add_argument("--out", help="결과표(markdown) 저장 경로, 예: logs/scenario_results.md")
    ap.add_argument("--replay-csv", help="합성 시나리오 대신 실측 CSV 구간을 그대로 보냄")
    ap.add_argument("--replay-end", nargs="+", default=[], help="재현할 구간의 마지막 날짜들")
    args = ap.parse_args()

    if args.replay_csv:
        summaries = []
        for end in args.replay_end:
            rows = replay_rows(args.replay_csv, end)
            inputs = f"실측 기준일 {rows[119]['date']}~{rows[146]['date']}"
            print(f"\n[replay {end}] {inputs}")
            result = send_batch(rows, f"replay ~{end}", args.url, args.timeout)
            print(
                f"  HTTP {result.get('http_status')}: {json.dumps(result.get('drift_check') or result.get('error'), ensure_ascii=False)[:600]}"
            )
            summaries.append(summarize(result, inputs))
        table = to_markdown(summaries, f"실측 재현 {args.replay_csv}, rows={BATCH_N}", "입력 구간")
        print("\n" + table)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(table + "\n")
        return

    base, sigma = base_from_csv(args.base_csv) if args.base_csv else (None, None)
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    source = (
        f"실측 기준({args.base_csv}) + 합성 충격" if args.base_csv else "합성 기준값 + 합성 충격"
    )
    meta = f"seed={args.seed}, rows={BATCH_N}, shock_day={SHOCK_DAY}, {source} - 성능 증빙 아님"
    print(meta)

    summaries = []
    for idx, name in enumerate(names):
        # 시나리오마다 서로 다른 기간의 배치로 보낸다. 같은 날짜로 보내면 서버는 같은 판정으로 보고
        # 재학습을 한 번만 한다(retrain_trigger 중복 방지) - 시나리오별 반응을 비교할 수 없게 된다.
        start = (date.fromisoformat(args.start) + timedelta(days=idx * BATCH_N)).isoformat()
        opts = dict(seed=args.seed, start=start, base=base, sigma=sigma)
        rows = generate_rows(name, **opts)
        inputs = describe(rows, generate_rows("normal", **opts))
        print(f"\n[{name}] {SCENARIOS[name]}\n  입력: {inputs}")
        if args.dry_run:
            summaries.append(summarize({"label": name, "http_status": "dry-run"}, inputs))
            continue
        result = send_batch(rows, name, args.url, args.timeout)
        print(
            f"  HTTP {result.get('http_status')}: {json.dumps(result.get('drift_check') or result.get('error'), ensure_ascii=False)[:400]}"
        )
        summaries.append(summarize(result, inputs))

    table = to_markdown(summaries, meta)
    print("\n" + table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"\n결과표 저장: {args.out}")


if __name__ == "__main__":
    main()
