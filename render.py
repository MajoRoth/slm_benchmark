import json
import os
from pathlib import Path

from jinja2 import FileSystemLoader, Environment


def render_page(index_path, description, sections):
    loader = FileSystemLoader(searchpath="./templates")
    env = Environment(loader=loader)
    template = env.get_template("demo.html.jinja2")

    html = template.render(
        page_title="Hebrew TTS",
        description=description,
        sections=sections,

    )

    if not os.path.exists(os.path.dirname(index_path)):
        os.makedirs(os.path.dirname(index_path))

    with open(index_path, 'w') as f:
        f.write(html)


def render_background():
    index_path = "/Users/amitroth/PycharmProjects/slm_benchmark/background/index.html"

    description = """
        background noises samples from FSD50K. we sampled recordings longer then 5 seconds and with only 1 leaf label.
    """

    dirs = list(Path("/Users/amitroth/PycharmProjects/slm_benchmark/audio/background_noises_samples").glob("*"))


    wavs = [
        [Path(os.path.relpath(p, os.path.dirname(index_path))) for p in dir.glob("*")] for dir in dirs
    ]

    background_sections = [
        {
            "title": f"FSD50K dataset samples",
            "description": f"sampled between all of the leaves in the ontology",
            "table": {
                "headers": ["class", "sample 1", "sample 2"],
                "content": [
                    [
                        f"{l[0].parts[-2]}",
                        f'"{str(l[0])}"',
                        f'"{str(l[1])}"' if len(l) >= 2 else "",

                    ] for l in wavs
                ]
            }
        }
    ]

    print(background_sections)

    render_page(index_path=index_path, description=description, sections=background_sections)



if __name__ == '__main__':
    render_background()