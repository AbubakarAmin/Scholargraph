"""
EditorAgent — submission-grade LaTeX, DOI bibliography, Limitations, companion repo.
Provides create_final_paper / generate_latex expected by main.py.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config
from core.utils import log_agent_action
from core.llm import call_llm, generate_embedding
from core.llm import get_llm_client
from core.context import RunContext, get_active_context
from core.contracts import ExperimentOutput, Paper, Plan, Topic
from core.memory import memory
from core.verification import extract_citation_ids, resolve_doi, resolve_arxiv


class EditorAgent:
    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()
        runtime_config = self.context.config if self.context else config
        self.vector_memory = self.context.memory if self.context else memory
        self.output_dir = runtime_config.output_dir
        self.companion_dir = runtime_config.companion_repo_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.companion_dir, exist_ok=True)

    # --- API expected by main.py ---

    def create_final_paper(
        self,
        topic: Topic,
        sections: Dict[str, str],
        plan: Plan,
        engineer_outputs: Optional[Dict[str, ExperimentOutput]] = None,
        debate_results: Optional[List[Any]] = None,
    ) -> Paper:
        sections = dict(sections)
        # Honest Limitations from unresolved Challenger objections
        if "Limitations" not in sections or len(sections.get("Limitations", "")) < 80:
            sections["Limitations"] = self._limitations_from_debate(debate_results, plan)

        bib_entries, bib_map = self._bibliography_from_dois(sections)
        companion = self._write_companion_repo(topic, plan, engineer_outputs or {})

        return {
            "topic": topic,
            "sections": sections,
            "plan": plan,
            "engineer_outputs": engineer_outputs or {},
            "bibliography": bib_entries,
            "bib_map": bib_map,
            "companion_repo": companion,
            "debate_results": debate_results or [],
            "timestamp": datetime.now().isoformat(),
        }

    def generate_latex(self, final_paper: Paper) -> str:
        topic = final_paper["topic"]
        sections = final_paper["sections"]
        plan = final_paper.get("plan") or {}
        bib = final_paper.get("bibliography") or ""
        latex = self._generate_latex_document(topic, sections, plan)
        # Attach resolved bib as comment + file write
        bib_path = os.path.join(self.output_dir, "references.bib")
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write(bib or "% no resolved citations\n")
        latex_path = os.path.join(
            self.output_dir, f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
        )
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        try:
            self.vector_memory.add_embedding(
                generate_embedding(topic.get("title", "")),
                {"type": "final_paper", "topic": topic.get("title"), "latex": latex_path},
            )
        except Exception:
            pass
        log_agent_action("EditorAgent", "latex_generated", {"path": latex_path})
        return latex

    def assemble_paper(
        self,
        topic: Topic,
        sections: Dict[str, str],
        plan: Plan,
        engineer_outputs: Optional[Dict[str, ExperimentOutput]] = None,
        debate_results: Optional[List[Any]] = None,
    ) -> Paper:
        final = self.create_final_paper(
            topic, sections, plan, engineer_outputs, debate_results
        )
        latex = self.generate_latex(final)
        return {
            "success": True,
            "final_paper": final,
            "latex": latex,
            "latex_file": os.path.join(self.output_dir, "paper_output.tex"),
            "bib_file": os.path.join(self.output_dir, "references.bib"),
            "companion_repo": final.get("companion_repo"),
            "timestamp": str(datetime.now()),
        }

    def _limitations_from_debate(
        self,
        debate_results: Optional[List[Any]],
        plan: Plan,
    ) -> str:
        objections = []
        for d in debate_results or []:
            unresolved = getattr(d, "unresolved_objections", None)
            if unresolved is None and isinstance(d, dict):
                unresolved = d.get("unresolved_objections") or []
            objections.extend(unresolved or [])
        if not objections:
            # Fall back to plan flags
            objections = (plan or {}).get("unfalsifiable_flags") or [
                "Results are based on synthetic or limited public data.",
                "Compute budget constrained sandbox experiments; scale-up unverified.",
            ]
        bullets = "\n".join(f"- {o}" for o in objections[:12])
        return (
            "# Limitations\n\n"
            "The following limitations are carried forward from adversarial debate "
            "and planning checks (unresolved objections and unverified assumptions):\n\n"
            f"{bullets}\n"
        )

    def _bibliography_from_dois(self, sections: Dict[str, str]) -> tuple:
        text = "\n".join(sections.values())
        ids = extract_citation_ids(text)
        entries = []
        bib_map = {}
        idx = 1
        for doi in ids["dois"]:
            info = resolve_doi(doi)
            if not info.get("resolved"):
                continue
            key = f"doi{idx:03d}"
            title = (info.get("title") or "Untitled").replace("{", "").replace("}", "")
            year = info.get("year") or "n.d."
            journal = (info.get("container") or "Journal").replace("{", "").replace("}", "")
            entries.append(
                f"@article{{{key},\n"
                f"  title={{{title}}},\n"
                f"  year={{{year}}},\n"
                f"  journal={{{journal}}},\n"
                f"  doi={{{doi}}}\n}}"
            )
            bib_map[doi] = key
            idx += 1
        for aid in ids["arxiv_ids"]:
            info = resolve_arxiv(aid)
            if not info.get("resolved"):
                continue
            key = f"arxiv{idx:03d}"
            title = (info.get("title") or aid).replace("{", "").replace("}", "")
            entries.append(
                f"@article{{{key},\n"
                f"  title={{{title}}},\n"
                f"  year={{20{aid[:2]}}},\n"
                f"  journal={{arXiv}},\n"
                f"  eprint={{{aid}}}\n}}"
            )
            bib_map[aid] = key
            idx += 1
        return "\n\n".join(entries), bib_map

    def _write_companion_repo(
        self,
        topic: Topic,
        plan: Plan,
        engineer_outputs: Dict[str, ExperimentOutput],
    ) -> Dict[str, str]:
        root = Path(self.companion_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "experiments").mkdir(exist_ok=True)

        req = "numpy\npandas\nscikit-learn\nmatplotlib\nscipy\n"
        (root / "requirements.txt").write_text(req, encoding="utf-8")

        readme = f"""# Companion code — {topic.get('title', 'Research')}

Auto-generated by ScholarGraph Editor.

## Setup
```bash
pip install -r requirements.txt
python run_experiments.py
```

## Experiments
See `experiments/` for code snapshots from the Engineer agent.
Raw multi-seed results live alongside the paper outputs.
"""
        (root / "README.md").write_text(readme, encoding="utf-8")

        runner_lines = [
            '"""Entry point for companion experiments."""',
            "import json",
            "from pathlib import Path",
            "",
            "def main():",
            "    print('Companion experiments')",
            "    for p in Path('experiments').glob('*.py'):",
            "        print(' -', p.name)",
            "",
            "if __name__ == '__main__':",
            "    main()",
        ]
        (root / "run_experiments.py").write_text("\n".join(runner_lines) + "\n", encoding="utf-8")

        paths = {"readme": str(root / "README.md"), "requirements": str(root / "requirements.txt")}
        for name, out in engineer_outputs.items():
            code = out.get("code") if isinstance(out, dict) else None
            if code:
                safe = re.sub(r"[^\w\-]+", "_", name)[:60]
                path = root / "experiments" / f"{safe}.py"
                path.write_text(code, encoding="utf-8")
                paths[safe] = str(path)
        return paths

    def _generate_latex_document(self, topic, sections, plan) -> str:
        abstract = self._extract_abstract(sections) if sections.get("Abstract") else topic.get("description", "")
        header = self._create_document_header(topic, plan, abstract)
        body_parts = []
        order = [
            "Introduction", "Related Work", "Methods", "Experiments",
            "Results", "Discussion", "Limitations", "Conclusion",
        ]
        seen = set()
        for name in order:
            if name in sections and name != "Abstract":
                body_parts.append(f"\\section{{{name}}}\n\n{self._process_section_content(sections[name], name)}\n")
                seen.add(name)
        for name, content in sections.items():
            if name not in seen and name != "Abstract":
                body_parts.append(f"\\section{{{name}}}\n\n{self._process_section_content(content, name)}\n")
        footer = self._create_document_footer()
        return header + "\n\n" + "\n\n".join(body_parts) + "\n\n" + footer

    def _create_document_header(self, topic, plan, abstract: str = "") -> str:
        abs_block = abstract.replace("&", "\\&") if abstract else "Abstract pending."
        return f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{graphicx,hyperref,geometry,natbib,booktabs}}
\\geometry{{margin=1in}}
\\title{{{topic.get('title', 'Research Paper')}}}
\\author{{ScholarGraph Research System}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle
\\begin{{abstract}}
{abs_block}
\\end{{abstract}}
"""

    def _create_document_footer(self) -> str:
        return "\\bibliographystyle{plain}\n\\bibliography{references}\n\\end{document}\n"

    def _process_section_content(self, content: str, section_name: str) -> str:
        content = re.sub(r"^#+\s*.*$", "", content, flags=re.MULTILINE)
        content = content.replace("&", "\\&").replace("%", "\\%")
        return content.strip()

    def _extract_abstract(self, sections: Dict[str, str]) -> str:
        if "Abstract" in sections:
            return re.sub(r"^#+\s*Abstract\s*", "", sections["Abstract"]).strip()
        return "Abstract to be added."
