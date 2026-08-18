from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def markdown_spilt(markdown: str) -> list[str]:
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    sections = markdown_splitter.split_text(markdown)

    # 第二步：对每个 section 做递归分割
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = recursive_splitter.split_documents(sections)

    return [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
