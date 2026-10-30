# services/document_parser.py

from docling.document_converter import DocumentConverter


class DocumentParser:

    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, file_path: str) -> str:
        """
        使用 Docling 解析文档并转换为 Markdown
        """
        result = self.converter.convert(file_path)

        document = result.document

        markdown_content = document.export_to_markdown()

        if not markdown_content.strip():
            raise ValueError("文档解析结果为空")

        return markdown_content