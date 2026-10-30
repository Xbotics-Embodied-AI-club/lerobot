"""LeRobot 训练与部署教程的 Sphinx 配置。"""

project = "LeRobot 训练与部署教程"
author = "Xbotics 具身智能社区"
language = "zh_CN"
root_doc = "index"
source_suffix = {".md": "markdown"}

extensions = ["myst_parser", "sphinx_copybutton"]
exclude_patterns = ["_build"]

myst_enable_extensions = ["attrs_inline", "colon_fence", "html_image"]
myst_heading_anchors = 3
copybutton_prompt_text = "$ "

html_theme = "sphinx_book_theme"
html_title = project
html_static_path = ["_static"]
html_css_files = ["site.css"]
html_context = {"default_mode": "auto"}
html_theme_options = {
    "repository_url": "https://github.com/Xbotics-Embodied-AI-club/lerobot",
    "repository_branch": "docs/github-pages",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "show_navbar_depth": 1,
}
