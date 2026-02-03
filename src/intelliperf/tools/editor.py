"""Code editing tools."""
from pathlib import Path
from typing import List
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import shutil


class EditFileInput(BaseModel):
    """Input for edit_file tool."""
    file_path: str = Field(description="Path to file to edit", alias="filename")
    old_content: str = Field(description="Exact text to replace (must be unique in file)", alias="old_string")
    new_content: str = Field(description="New text to insert", alias="new_string")


def edit_file_impl(file_path: str, old_content: str, new_content: str) -> str:
    """Edit file using search and replace."""
    try:
        path = Path(file_path)
        if not path.exists():
            return (f"Error: File '{file_path}' not found\n"
                   f"Tip: Use 'shell' with 'ls' to see available files, or 'cat' to check file names.")

        # Check if old and new content are identical
        if old_content == new_content:
            return (f"Warning: old_content and new_content are identical. No changes made to '{file_path}'.\n"
                   f"Tip: Make sure you're actually changing something in the file.")

        with open(path, 'r') as f:
            content = f.read()

        if old_content not in content:
            return (f"Error: The old_content string was not found in '{file_path}'\n"
                   f"Tip: Use 'shell' with 'cat {file_path}' to see the actual file content, then provide exact matching text.")

        count = content.count(old_content)
        if count > 1:
            return (f"Error: old_content appears {count} times in '{file_path}'. Must be unique for safe editing.\n"
                   f"Tip: Include more surrounding context to make it unique.")

        new_file_content = content.replace(old_content, new_content)

        with open(path, 'w') as f:
            f.write(new_file_content)

        return f"Successfully edited {file_path}"
    except Exception as e:
        return f"Error editing {file_path}: {e}"


def create_editor_tools(work_dir: Path) -> List[StructuredTool]:
    """Create code editing tools."""
    return [
        StructuredTool.from_function(
            func=lambda file_path="", filename="", old_content="", old_string="", new_content="", new_string="", **kwargs: edit_file_impl(
                file_path or filename,
                old_content or old_string,
                new_content or new_string
            ),
            name="edit_file",
            description="Edit a file by replacing old_content with new_content. The old_content must be exact and unique.",
            args_schema=EditFileInput
        ),
    ]
