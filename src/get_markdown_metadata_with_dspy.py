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
    uv run python src/get_markdown_metadata_with_dspy.py --prompts configs/prompts01.toml --llm 'openrouter/openai/gpt-oss-120b'
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import dspy
import tomllib
import typer
from rich.console import Console

from models.markdown import MarkdownDocument

app = typer.Typer()
console = Console()


class GenerateMarkdown(dspy.Signature):
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
    prompts: Path = typer.Option(
        ..., help="システムプロンプトとユーザープロンプトが記載されたファイルのパス"
    ),
    config: Path = typer.Option("configs/config.toml", help="設定ファイルのパス"),
    llm: str = typer.Option(
        "openrouter/openai/gpt-oss-120b",
        help="LLM モデル名（例: openrouter/openai/gpt-oss-120b）",
    ),
) -> None:
    """構造化されたマークダウン文書を生成します（DSPy版）。"""
    with open(prompts, "rb") as f:
        prompts = tomllib.load(f)
    system_prompt = prompts["prompt"]["system"]
    user_prompt = prompts["prompt"]["user"]

    # GenerateMarkdown の docstring を動的に設定
    GenerateMarkdown.__doc__ = system_prompt

    with open(config, "rb") as f:
        config = tomllib.load(f)
    temperature = config["temperature"]

    # OpenRouter の API キーを環境変数から取得
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        typer.echo("エラー: OPENROUTER_API_KEY 環境変数が設定されていません", err=True)
        raise typer.Exit(1)

    # DSPy の LiteLLM モデルを設定
    lm = dspy.LM(
        model=llm,
        api_key=api_key,
        temperature=temperature,
        cache=False,  # キャッシュを無効化
    )
    dspy.configure(lm=lm)

    # マークダウン生成モジュールを作成
    generator = MarkdownGenerator()

    # マークダウンを生成
    try:
        with console.status("[bold green]LLM 回答中...", spinner="dots") as status:
            result = generator(prompt=user_prompt)
            status.update("[bold green]生成完了！")

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
