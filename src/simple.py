"""LLM 名をプロンプトを受け取ってレスポンスを出力する。

Example:
    uv run python src/simple.py --llm 'openrouter/openai/gpt-oss-120b' --prompt '9.11 と 9.9 はどちらが大きい？'
"""

import os

import typer
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel

app = typer.Typer()


@app.command()
async def main(
    llm: str = typer.Option(
        ..., help="LLM モデル名（例: openrouter/openai/gpt-oss-120b）"
    ),
    prompt: str = typer.Option(..., help="送信するプロンプト"),
) -> None:
    """LLM にプロンプトを送信してレスポンスを取得します。"""
    # OpenRouter の API キーを環境変数から取得
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        typer.echo("エラー: OPENROUTER_API_KEY 環境変数が設定されていません", err=True)
        raise typer.Exit(1)

    # LiteLLM モデルを使用してエージェントを作成
    agent = Agent(
        name="Assistant",
        instructions="ユーザーの質問に対して、正確で分かりやすい回答を提供してください。",
        model=LitellmModel(model=llm, api_key=api_key),
    )

    # エージェントを実行
    result = await Runner.run(agent, prompt)

    # 結果を出力
    typer.echo(result.final_output)


if __name__ == "__main__":
    app()
