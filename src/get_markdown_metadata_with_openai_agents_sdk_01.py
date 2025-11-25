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
    uv run python src/get_markdown.py --prompt '1000 文字程度の記事を書いてください。テーマは、生成 AI です。' --llm 'openrouter/openai/gpt-oss-120b'

"""

from __future__ import annotations

import asyncio
import os
from typing import Union

import typer
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from pydantic import BaseModel, Field

app = typer.Typer()


class Content(BaseModel):
    """コンテンツ"""

    content: str = Field(
        ...,
        description="本文のテキスト",
        examples=["ここでは導入をおこなう。", "メインコンテンツを扱う。"],
    )


class Heading(BaseModel):
    """見出しのメタデータ"""

    level: int = Field(
        ...,
        description="見出しのレベル（例: 1, 2, 3）",
        examples=[2, 3],
    )
    text: str = Field(
        ...,
        description="見出しのテキスト",
        examples=["導入", "まとめ"],
    )
    children: list[Union[Content, Heading]] = Field(
        ...,
        description="子要素のリスト",
        examples=[
            [{"content": "ここでは導入をおこなう。"}],
            [
                {"content": "メインコンテンツを扱う。"},
                {
                    "level": 3,
                    "text": "サブコンテンツ 1",
                    "children": [{"content": "サブコンテンツ 1 を扱う。"}],
                },
            ],
        ],
    )


class MarkdownDocument(BaseModel):
    """マークダウン文書の構造を定義します。"""

    contents: list[Union[Content, Heading]] = Field(
        ...,
        description="本文のリスト",
        examples=[
            [
                {
                    "level": 2,
                    "text": "導入",
                    "children": [{"content": "ここでは導入をおこなう。"}],
                },
                {
                    "level": 2,
                    "text": "メインコンテンツ",
                    "children": [
                        {"content": "メインコンテンツを扱う。"},
                        {
                            "level": 3,
                            "text": "サブコンテンツ 1",
                            "children": [{"content": "サブコンテンツ 1 を扱う。"}],
                        },
                        {
                            "level": 3,
                            "text": "サブコンテンツ 2",
                            "children": [{"content": "サブコンテンツ 2 を扱う。"}],
                        },
                    ],
                },
                {
                    "level": 2,
                    "text": "まとめ",
                    "children": [{"content": "ここではまとめをおこなう。"}],
                },
            ]
        ],
    )


@app.command()
def main(
    prompt: str = typer.Option(..., help="マークダウン文書を生成するためのプロンプト"),
    llm: str = typer.Option(
        "openrouter/openai/gpt-oss-120b",
        help="LLM モデル名（例: openrouter/openai/gpt-oss-120b）",
    ),
) -> None:
    """構造化されたマークダウン文書を生成します。"""
    asyncio.run(run_async(llm, prompt))


async def run_async(llm: str, prompt: str) -> None:
    """非同期でエージェントを実行します。"""
    # OpenRouter の API キーを環境変数から取得
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        typer.echo("エラー: OPENROUTER_API_KEY 環境変数が設定されていません", err=True)
        raise typer.Exit(1)

    # LiteLLM モデルを使用してエージェントを作成
    agent = Agent(
        name="Markdown Generator",
        instructions=(
            "ユーザーの要求に基づいて、構造化されたマークダウン文書を生成してください。"
            "文書は見出し（Heading）と本文（Content）で構成され、見出しは階層構造を持つことができます。"
            "見出しのレベル（1-6）を適切に設定し、各セクションに適切な本文を配置してください。"
            "見出しには子要素として、さらに下位の見出しや本文を含めることができます。"
        ),
        model=LitellmModel(model=llm, api_key=api_key),
        output_type=MarkdownDocument,
    )

    # エージェントを実行
    result = await Runner.run(agent, prompt)

    # 構造化された出力を取得
    document = result.final_output_as(MarkdownDocument)

    # JSON 形式で出力（見出しのレベルなどのメタデータを含む）
    output = document.model_dump_json(indent=2, ensure_ascii=False)
    typer.echo(output)


if __name__ == "__main__":
    app()
