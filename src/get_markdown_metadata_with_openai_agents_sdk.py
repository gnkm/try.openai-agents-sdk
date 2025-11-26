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
    uv run python src/get_markdown_metadata_with_openai_agents_sdk.py --prompts configs/prompts01.toml --llm 'openrouter/openai/gpt-oss-120b'

"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import tomllib
import typer
from agents import Agent, ModelSettings, Runner
from agents.extensions.models.litellm_model import LitellmModel
from rich.console import Console

from models.markdown import MarkdownDocument

app = typer.Typer()
console = Console()


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
) -> None:
    """構造化されたマークダウン文書を生成します（OpenAI Agents SDK版）。"""
    with open(prompts, "rb") as f:
        prompts_data = tomllib.load(f)
    system_prompt = prompts_data["prompt"]["system"]
    user_prompt = prompts_data["prompt"]["user"]

    with open(config, "rb") as f:
        config_data = tomllib.load(f)
    temperature = config_data["temperature"]

    asyncio.run(run_async(llm, system_prompt, user_prompt, temperature))


async def run_async(
    llm: str, system_prompt: str, user_prompt: str, temperature: float
) -> None:
    """非同期でエージェントを実行します。"""
    # OpenRouter の API キーを環境変数から取得
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        typer.echo("エラー: OPENROUTER_API_KEY 環境変数が設定されていません", err=True)
        raise typer.Exit(1)

    # LiteLLM モデルを使用してエージェントを作成
    agent = Agent(
        name="Markdown Generator",
        instructions=system_prompt,
        model=LitellmModel(model=llm, api_key=api_key),
        model_settings=ModelSettings(temperature=temperature),
        output_type=MarkdownDocument,
    )

    # エージェントを実行
    with console.status("[bold green]LLM 回答中...", spinner="dots") as status:
        result = await Runner.run(agent, user_prompt)
        status.update("[bold green]生成完了！")

    # 構造化された出力を取得
    document = result.final_output_as(MarkdownDocument)

    # JSON 形式で出力（見出しのレベルなどのメタデータを含む）
    output = document.model_dump_json(indent=2, ensure_ascii=False)
    typer.echo(output)


if __name__ == "__main__":
    app()
