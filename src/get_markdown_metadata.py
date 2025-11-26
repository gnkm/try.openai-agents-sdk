"""構造化したマークダウンデータを生成する。

マークダウンの[コンテンツ]に対して、[構造]を返す。

# コンテンツ

```
## 導入

ここでは導入をおこなう。

## メインコンテンツ

メインコンテンツを扱う。

### サブコンテンツ 1

サブコンテンツ 1 を扱う。

### サブコンテンツ 2

サブコンテンツ 2 を扱う。

## まとめ

ここではまとめをおこなう。
```

# 構造

```
{
    'contents': [
        {
            'level': 2,
            'text': '導入',
            'children': [
                {
                    'content': 'ここでは導入をおこなう。'
                },
            ],
        },
        {
            'level': 2,
            'text': 'メインコンテンツ',
            'children': [
                {'content': 'メインコンテンツを扱う。'},
                {'level': 3, 'text': 'サブコンテンツ 1', 'children': [{'content': 'サブコンテンツ 1 を扱う。'}]},
                {'level': 3, 'text': 'サブコンテンツ 2', 'children': [{'content': 'サブコンテンツ 2 を扱う。'}]},
            ],
        },
        {
            'level': 2,
            'text': 'まとめ',
            'children': [
                {
                    'content': 'ここではまとめをおこなう。'
                },
            ],

        }
}
```

Example:
    # OpenAI Agents SDK を使用（デフォルト）
    uv run python src/get_markdown_metadata.py --prompts configs/prompts01.toml

    # DSPy を使用
    uv run python src/get_markdown_metadata.py --prompts configs/prompts01.toml --backend dspy

"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import tomllib
import typer

from backends.dspy import run_with_dspy
from backends.openai_agents_sdk import run_with_openai_agents

app = typer.Typer()


@app.command()
def main(
    prompts: Path = typer.Option(
        ..., help="システムプロンプトとユーザープロンプトが記載されたファイルのパス"
    ),
    config: Path = typer.Option("configs/config.toml", help="設定ファイルのパス"),
    llm: str = typer.Option(
        "openrouter/openai/gpt-oss-120b",
        help="LLM モデル名（例: openrouter/openai/gpt-oss-120b）",
    ),
    backend: str = typer.Option(
        "openai-agents",
        help="バックエンド（'dspy' or 'openai-agents'）",
    ),
) -> None:
    """構造化されたマークダウン文書を生成します。"""
    # OpenRouter の API キーを環境変数から取得（早期検証）
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        typer.echo("エラー: OPENROUTER_API_KEY 環境変数が設定されていません", err=True)
        raise typer.Exit(1)

    with open(prompts, "rb") as f:
        prompts_data = tomllib.load(f)
    system_prompt = prompts_data["prompt"]["system"]
    user_prompt = prompts_data["prompt"]["user"]

    with open(config, "rb") as f:
        config_data = tomllib.load(f)
    temperature = config_data["temperature"]

    # バックエンドに応じて処理を分岐
    if backend == "dspy":
        run_with_dspy(llm, system_prompt, user_prompt, temperature, api_key)
    elif backend == "openai-agents":
        asyncio.run(
            run_with_openai_agents(
                llm, system_prompt, user_prompt, temperature, api_key
            )
        )
    else:
        typer.echo(
            f"エラー: 不正なバックエンド '{backend}'。'dspy' または 'openai-agents' を指定してください。",
            err=True,
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
