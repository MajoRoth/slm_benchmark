import json
import os
from pathlib import Path

from jinja2 import FileSystemLoader, Environment


def render_page(name, description, sections):
    loader = FileSystemLoader(searchpath="./templates")
    env = Environment(loader=loader)
    template = env.get_template("demo.html.jinja2")

    html = template.render(
        page_title="Hebrew TTS",
        description=description,
        sections=sections,

    )

    dir_name = Path(__file__).parent / name

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(dir_name / "index.html", 'w') as f:
        f.write(html)


def render_background():
    description = """
        background noises samples from FSD50K. we sampled recordings longer then 5 seconds and with only 1 leaf label.
    """

    dirs = list(Path("/Users/amitroth/PycharmProjects/slm_benchmark/audio/background_noises_samples").glob("*"))


    wavs = [
        [Path("../") / Path(os.path.relpath(p, os.path.dirname(__file__))) for p in dir.glob("*")] for dir in dirs
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

    render_page(name='background', description=description, sections=background_sections)


def render_emotions():
    pass

if __name__ == '__main__':
    render_background()