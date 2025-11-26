"""OpenAI Agents SDK バックエンドの実装"""

from __future__ import annotations

import typer
from agents import Agent, ModelSettings, Runner
from agents.extensions.models.litellm_model import LitellmModel
from rich.console import Console

from models.markdown import MarkdownDocument

console = Console()


async def run_with_openai_agents(
    llm: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    api_key: str,
) -> None:
    """OpenAI Agents SDK を使用してマークダウン文書を生成します。"""
    # LiteLLM モデルを使用してエージェントを作成
    agent = Agent(
        name="Markdown Generator",
        instructions=system_prompt,
        model=LitellmModel(model=llm, api_key=api_key),
        model_settings=ModelSettings(temperature=temperature),
        output_type=MarkdownDocument,
    )

    # エージェントを実行
    with console.status("[bold green]LLM 回答中...", spinner="dots"):
        result = await Runner.run(agent, user_prompt)

    # 構造化された出力を取得
    document = result.final_output_as(MarkdownDocument)

    # JSON 形式で出力（見出しのレベルなどのメタデータを含む）
    output = document.model_dump_json(indent=2, ensure_ascii=False)
    typer.echo(output)

