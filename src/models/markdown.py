"""Markdown ドキュメントの型を定める。"""

from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field


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
