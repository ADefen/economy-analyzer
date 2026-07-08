import click
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8")
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError("Поддерживаются только .csv, .xlsx, .xls")


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)


def generate_reason_and_recommendation(
    item_name: str,
    current_price: float,
    median_price: float,
    iqr_range: tuple[float, float],
    rarity: str,
    client: Optional[OpenAI] = None
) -> tuple[str, str]:

    if client is not None:
        prompt = (
            f"Ты — геймдизайнер. Проанализируй цену предмета в игре.\n"
            f"Предмет: {item_name}, редкость: {rarity}\n"
            f"Текущая цена: {current_price}\n"
            f"Медиана по категории: {median_price}\n"
            f"Рекомендуемый диапазон (IQR): {iqr_range[0]}–{iqr_range[1]}\n"
            f"Кратко (1–2 предложения) объясни, почему цена подозрительна.\n"
            f"Затем дай рекомендацию по цене и обоснование (1 предложение).\n"
            f"Отвечай кратко, без лишних слов, на русском."
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            text = resp.choices[0].message.content.strip()
            parts = text.split("\n")
            reason = parts[0] if len(parts) > 0 else "Аномалия по статистике."
            recommendation = parts[1] if len(parts) > 1 else f"Рекомендуемый диапазон: {iqr_range[0]}–{iqr_range[1]}"
            return reason, recommendation
        except Exception:
            pass

    if current_price < iqr_range[0]:
        reason = "Цена ниже нижней границы IQR — подозрительно низкая."
        recommendation = f"Рекомендуемый диапазон: {iqr_range[0]}–{iqr_range[1]}. Рассмотреть повышение."
    else:
        reason = "Цена выше верхней границы IQR — подозрительно высокая."
        recommendation = f"Рекомендуемый диапазон: {iqr_range[0]}–{iqr_range[1]}. Рассмотреть снижение."
    return reason, recommendation


@click.command()
@click.option("--input", required=True, help="Путь к файлу с данными (CSV/XLSX)")
@click.option("--output", required=True, help="Путь для выходного Excel-отчёта")
@click.option("--openai-api-key", default=None, help="API ключ OpenAI (опционально)")
@click.option("--price-column", default="price", help="Название колонки с ценой")
@click.option("--item-column", default="item_name", help="Название колонки с именем предмета")
@click.option("--rarity-column", default="rarity", help="Название колонки с редкостью")
def analyze_economy(
    input: str,
    output: str,
    openai_api_key: Optional[str],
    price_column: str,
    item_column: str,
    rarity_column: str
):
    df = load_data(input)

    if price_column not in df.columns or item_column not in df.columns:
        raise ValueError(f"В файле должны быть колонки: {price_column}, {item_column}")

    df = df.dropna(subset=[price_column])
    df[price_column] = pd.to_numeric(df[price_column], errors="coerce").dropna()
    df = df[df[price_column] > 0]

    client = None
    if openai_api_key and OPENAI_AVAILABLE:
        client = OpenAI(api_key=openai_api_key)

    results = []

    grouped = df.groupby(rarity_column, dropna=False)
    for rarity, group in grouped:
        prices = group[price_column]
        if len(prices) < 5:
            continue

        lower_iqr = prices.quantile(0.25) - 1.5 * (prices.quantile(0.75) - prices.quantile(0.25))
        upper_iqr = prices.quantile(0.75) + 1.5 * (prices.quantile(0.75) - prices.quantile(0.25))
        iqr_range = (max(0, lower_iqr), upper_iqr)
        median_price = prices.median()

        outliers = detect_outliers_iqr(prices)
        for _, row in group[outliers].iterrows():
            reason, recommendation = generate_reason_and_recommendation(
                item_name=row[item_column],
                current_price=row[price_column],
                median_price=median_price,
                iqr_range=iqr_range,
                rarity=str(rarity),
                client=client
            )
            results.append({
                "item_name": row[item_column],
                "rarity": str(rarity),
                "current_price": row[price_column],
                "median_price": median_price,
                "recommended_range": f"{iqr_range[0]:.2f}–{iqr_range[1]:.2f}",
                "reason": reason,
                "recommendation": recommendation,
                "risk_level": "HIGH" if (row[price_column] < iqr_range[0] * 0.5 or row[price_column] > iqr_range[1] * 1.5) else "MEDIUM"
            })

    result_df = pd.DataFrame(results)
    result_df.to_excel(output, index=False)

    click.echo(f"Готово: найдено {len(result_df)} аномалий. Отчёт сохранён в {output}")


if __name__ == "__main__":
    analyze_economy()
