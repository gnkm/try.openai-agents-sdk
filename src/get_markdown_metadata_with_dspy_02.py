"""
構造化したマークダウンデータを生成する。

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
    uv run python src/get_markdown_metadata_with_dspy_02.py --prompt '1000 文字程度の記事を書いてください。テーマは、生成 AI です。' --llm 'openrouter/openai/gpt-oss-120b'
"""

from __future__ import annotations

import json
import os
from typing import Union

import dspy
import typer
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


class GenerateMarkdown(dspy.Signature):
    """ユーザーの要求に基づいて、構造化されたマークダウン文書を生成します。

    文書は見出し（Heading）と本文（Content）で構成され、見出しは階層構造を持つことができます。
    見出しのレベル（1-6）を適切に設定し、各セクションに適切な本文を配置してください。
    見出しには子要素として、さらに下位の見出しや本文を含めることができます。

    出力は必ず JSON 形式で、以下の構造に従ってください：
    {
        "contents": [
            {
                "level": 2,
                "text": "見出しテキスト",
                "children": [
                    {"content": "本文テキスト"},
                    {
                        "level": 3,
                        "text": "サブ見出し",
                        "children": [{"content": "サブセクションの本文"}]
                    }
                ]
            }
        ]
    }
    """

    prompt: str = dspy.InputField(desc="マークダウン文書を生成するためのプロンプト")
    markdown_json: str = dspy.OutputField(
        desc="構造化されたマークダウン文書のJSON形式の文字列"
    )


class MarkdownGenerator(dspy.Module):
    """構造化されたマークダウン文書を生成するモジュール"""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateMarkdown)

    def forward(self, prompt: str) -> dspy.Prediction:
        """プロンプトから構造化されたマークダウン文書を生成します。"""
        prediction = self.generate(prompt=prompt)
        return prediction


@app.command()
def main(
    prompt: str = typer.Option(..., help="マークダウン文書を生成するためのプロンプト"),
    llm: str = typer.Option(
        "openrouter/openai/gpt-oss-120b",
        help="LLM モデル名（例: openrouter/openai/gpt-oss-120b）",
    ),
) -> None:
    """構造化されたマークダウン文書を生成します（DSPy版）。"""
    # OpenRouter の API キーを環境変数から取得
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        typer.echo("エラー: OPENROUTER_API_KEY 環境変数が設定されていません", err=True)
        raise typer.Exit(1)

    # DSPy の LiteLLM モデルを設定
    lm = dspy.LM(
        model=llm,
        api_key=api_key,
        temperature=0.7,
    )
    dspy.configure(lm=lm)

    # マークダウン生成モジュールを作成
    generator = MarkdownGenerator()

    # マークダウンを生成
    try:
        result = generator(prompt=prompt)

        # JSON文字列をパースしてバリデーション
        markdown_json_str = result.markdown_json.strip()

        # コードブロックで囲まれている場合は取り除く
        if markdown_json_str.startswith("```"):
            lines = markdown_json_str.split("\n")
            # 最初と最後の```行を除去
            markdown_json_str = "\n".join(
                line
                for i, line in enumerate(lines)
                if not line.strip().startswith("```")
            )

        # JSONとしてパース
        markdown_dict = json.loads(markdown_json_str)

        # Pydantic モデルでバリデーション
        document = MarkdownDocument(**markdown_dict)

        # JSON 形式で出力
        output = document.model_dump_json(indent=2, ensure_ascii=False)
        typer.echo(output)

    except json.JSONDecodeError as e:
        typer.echo(f"エラー: JSON のパースに失敗しました: {e}", err=True)
        typer.echo(f"生成された出力:\n{result.markdown_json}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
