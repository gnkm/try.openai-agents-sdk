"""DSPy バックエンドの実装"""

from __future__ import annotations

import json

import dspy
import typer
from rich.console import Console

from models.markdown import MarkdownDocument

console = Console()


class GenerateMarkdown(dspy.Signature):
    """マークダウン文書を生成するシグネチャ"""

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


def run_with_dspy(
    llm: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    api_key: str,
) -> None:
    """DSPy を使用してマークダウン文書を生成します。"""
    # GenerateMarkdown の docstring を動的に設定
    GenerateMarkdown.__doc__ = system_prompt

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
        with console.status("[bold green]LLM 回答中...", spinner="dots"):
            result = generator(prompt=user_prompt)

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

