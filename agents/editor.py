"""
EditorAgent - Assembles final LaTeX documents with proper formatting.
Handles bibliography, figures, and publication-ready output.
"""

import json
import os
from typing import Dict, Any, List
from datetime import datetime
import subprocess

from core.config import config
from core.utils import setup_gemini, call_gemini, log_agent_action
from core.memory import memory

class EditorAgent:
    """Agent for assembling final research papers in LaTeX format."""
    
    def __init__(self):
        self.gemini_client = setup_gemini()
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def assemble_paper(self, topic: Dict[str, Any], sections: Dict[str, str], 
                      plan: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble the final research paper."""
        log_agent_action("EditorAgent", "start_assembly", {"topic": topic['title']})
        
        try:
            # Generate LaTeX document
            latex_content = self._generate_latex_document(topic, sections, plan)
            
            # Create bibliography
            bibliography = self._generate_bibliography(topic, sections)
            
            # Add figures and tables
            figures = self._add_figures_and_tables(sections)
            
            # Compile final document
            final_document = self._compile_final_document(latex_content, bibliography, figures)
            
            # Generate output files
            output_files = self._generate_output_files(final_document, topic)
            
            # Store in memory
            self._store_paper(final_document, output_files, topic)
            
            log_agent_action("EditorAgent", "assembly_complete", {
                "topic": topic['title'],
                "output_files": list(output_files.keys())
            })
            
            return {
                'latex_file': output_files.get('latex_file'),
                'pdf_file': output_files.get('pdf_file'),
                'bib_file': output_files.get('bib_file'),
                'zip_file': output_files.get('zip_file'),
                'success': True,
                'timestamp': str(datetime.now())
            }
            
        except Exception as e:
            log_agent_action("EditorAgent", "assembly_error", {"error": str(e)})
            return {
                'error': str(e),
                'success': False,
                'timestamp': str(datetime.now())
            }
    
    def _generate_latex_document(self, topic: Dict[str, Any], sections: Dict[str, str], 
                                plan: Dict[str, Any]) -> str:
        """Generate the main LaTeX document."""
        # Create document header
        header = self._create_document_header(topic, plan)
        
        # Process sections
        processed_sections = []
        for section_name, content in sections.items():
            processed_content = self._process_section_content(content, section_name)
            processed_sections.append(processed_content)
        
        # Create document body
        body = "\n\n".join(processed_sections)
        
        # Create document footer
        footer = self._create_document_footer()
        
        return header + "\n\n" + body + "\n\n" + footer
    
    def _create_document_header(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """Create the LaTeX document header."""
        return f"""\\documentclass[11pt,a4paper]{{article}}

% Packages
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage{{geometry}}
\\usepackage{{natbib}}
\\usepackage{{algorithm}}
\\usepackage{{algorithmic}}
\\usepackage{{booktabs}}
\\usepackage{{multirow}}
\\usepackage{{array}}

% Page setup
\\geometry{{margin=1in}}

% Hyperref setup
\\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    citecolor=blue
}}

% Title and author
\\title{{{topic['title']}}}
\\author{{AI Research System}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{self._extract_abstract(sections) if 'Abstract' in sections else 'Abstract to be added.'}
\\end{{abstract}}

\\tableofcontents
\\newpage
"""
    
    def _create_document_footer(self) -> str:
        """Create the LaTeX document footer."""
        return """
\\bibliographystyle{plain}
\\bibliography{references}

\\end{document}
"""
    
    def _process_section_content(self, content: str, section_name: str) -> str:
        """Process section content for LaTeX formatting."""
        # Remove markdown headers
        content = content.replace('# ', '\\section{').replace('#', '}')
        content = content.replace('## ', '\\subsection{').replace('##', '}')
        content = content.replace('### ', '\\subsubsection{').replace('###', '}')
        
        # Convert code blocks
        content = self._convert_code_blocks(content)
        
        # Convert inline code
        content = self._convert_inline_code(content)
        
        # Convert lists
        content = self._convert_lists(content)
        
        # Convert emphasis
        content = content.replace('**', '\\textbf{').replace('**', '}')
        content = content.replace('*', '\\textit{').replace('*', '}')
        
        return content
    
    def _convert_code_blocks(self, content: str) -> str:
        """Convert markdown code blocks to LaTeX."""
        import re
        
        # Find code blocks
        pattern = r'```(\w+)?\n(.*?)\n```'
        
        def replace_code_block(match):
            lang = match.group(1) or 'text'
            code = match.group(2)
            
            # Escape LaTeX special characters
            code = code.replace('\\', '\\textbackslash{}')
            code = code.replace('{', '\\{')
            code = code.replace('}', '\\}')
            code = code.replace('^', '\\textasciicircum{}')
            code = code.replace('_', '\\_')
            code = code.replace('%', '\\%')
            code = code.replace('$', '\\$')
            code = code.replace('#', '\\#')
            code = code.replace('&', '\\&')
            
            return f'\\begin{{verbatim}}\n{code}\n\\end{{verbatim}}'
        
        return re.sub(pattern, replace_code_block, content, flags=re.DOTALL)
    
    def _convert_inline_code(self, content: str) -> str:
        """Convert inline code to LaTeX."""
        import re
        
        # Find inline code
        pattern = r'`([^`]+)`'
        
        def replace_inline_code(match):
            code = match.group(1)
            # Escape LaTeX special characters
            code = code.replace('\\', '\\textbackslash{}')
            code = code.replace('{', '\\{')
            code = code.replace('}', '\\}')
            code = code.replace('_', '\\_')
            return f'\\texttt{{{code}}}'
        
        return re.sub(pattern, replace_inline_code, content)
    
    def _convert_lists(self, content: str) -> str:
        """Convert markdown lists to LaTeX."""
        lines = content.split('\n')
        in_list = False
        result = []
        
        for line in lines:
            if line.strip().startswith('- '):
                if not in_list:
                    result.append('\\begin{itemize}')
                    in_list = True
                item = line.strip()[2:]  # Remove '- '
                result.append(f'\\item {item}')
            elif line.strip().startswith('1. '):
                if not in_list:
                    result.append('\\begin{enumerate}')
                    in_list = True
                item = line.strip()[3:]  # Remove '1. '
                result.append(f'\\item {item}')
            else:
                if in_list:
                    result.append('\\end{itemize}' if '- ' in content else '\\end{enumerate}')
                    in_list = False
                result.append(line)
        
        if in_list:
            result.append('\\end{itemize}' if '- ' in content else '\\end{enumerate}')
        
        return '\n'.join(result)
    
    def _extract_abstract(self, sections: Dict[str, str]) -> str:
        """Extract abstract content from sections."""
        if 'Abstract' in sections:
            abstract = sections['Abstract']
            # Remove markdown formatting
            abstract = abstract.replace('# Abstract', '').strip()
            return abstract
        return "Abstract to be added."
    
    def _generate_bibliography(self, topic: Dict[str, Any], sections: Dict[str, str]) -> str:
        """Generate bibliography entries."""
        # Extract citations from all sections
        all_citations = []
        for section_name, content in sections.items():
            citations = self._extract_citations_from_text(content)
            all_citations.extend(citations)
        
        # Remove duplicates
        unique_citations = list(set(all_citations))
        
        # Generate BibTeX entries
        bibtex_entries = []
        for i, citation in enumerate(unique_citations):
            bibtex_entry = self._generate_bibtex_entry(citation, i + 1)
            bibtex_entries.append(bibtex_entry)
        
        return "\n\n".join(bibtex_entries)
    
    def _extract_citations_from_text(self, text: str) -> List[str]:
        """Extract citation patterns from text."""
        import re
        
        # Common citation patterns
        patterns = [
            r'\[([^\]]+)\]',  # [Author et al., 2023]
            r'\(([^)]+)\)',   # (Author et al., 2023)
            r'Author et al\.\s+\d{4}',  # Author et al. 2023
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return list(set(citations))
    
    def _generate_bibtex_entry(self, citation: str, index: int) -> str:
        """Generate a BibTeX entry for a citation."""
        # Generate a key based on the citation
        key = f"ref{index:03d}"
        
        # Try to extract author and year from citation
        import re
        year_match = re.search(r'\d{4}', citation)
        year = year_match.group() if year_match else "2023"
        
        author = citation.replace(year, '').strip()
        if author.endswith(','):
            author = author[:-1]
        
        return f"""@article{{{key},
  title={{{citation}}},
  author={{{author}}},
  year={{{year}}},
  journal={{arXiv preprint}},
  doi={{10.1000/000000}}
}}"""
    
    def _add_figures_and_tables(self, sections: Dict[str, str]) -> List[str]:
        """Add figures and tables to the document."""
        figures = []
        
        # Look for figure references in sections
        for section_name, content in sections.items():
            if 'figure' in content.lower() or 'plot' in content.lower():
                # Generate a figure placeholder
                figure_path = f"figures/{section_name.lower().replace(' ', '_')}.png"
                figures.append(figure_path)
        
        return figures
    
    def _compile_final_document(self, latex_content: str, bibliography: str, 
                               figures: List[str]) -> str:
        """Compile the final document with all components."""
        # Add bibliography to the document
        if bibliography:
            # Replace bibliography placeholder
            latex_content = latex_content.replace('\\bibliography{references}', 
                                               f'\\bibliography{{references}}\n\n% Bibliography entries:\n{bibliography}')
        
        # Add figure includes
        for figure in figures:
            if os.path.exists(figure):
                latex_content = latex_content.replace('\\end{document}', 
                                                   f'\\includegraphics[width=0.8\\textwidth]{{{figure}}}\n\\end{{document}}')
        
        return latex_content
    
    def _generate_output_files(self, final_document: str, topic: Dict[str, Any]) -> Dict[str, str]:
        """Generate all output files."""
        output_files = {}
        
        # Generate LaTeX file
        latex_filename = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
        latex_path = os.path.join(self.output_dir, latex_filename)
        
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(final_document)
        
        output_files['latex_file'] = latex_path
        
        # Generate bibliography file
        bib_filename = "references.bib"
        bib_path = os.path.join(self.output_dir, bib_filename)
        
        # Extract bibliography from document
        bib_start = final_document.find('% Bibliography entries:')
        if bib_start != -1:
            bib_content = final_document[bib_start:].replace('% Bibliography entries:', '').strip()
            with open(bib_path, 'w', encoding='utf-8') as f:
                f.write(bib_content)
            output_files['bib_file'] = bib_path
        
        # Try to compile PDF
        try:
            pdf_filename = latex_filename.replace('.tex', '.pdf')
            pdf_path = os.path.join(self.output_dir, pdf_filename)
            
            # Run pdflatex
            result = subprocess.run([
                'pdflatex', '-interaction=nonstopmode', '-output-directory', self.output_dir, latex_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(pdf_path):
                output_files['pdf_file'] = pdf_path
            else:
                log_agent_action("EditorAgent", "pdf_compilation_failed", {
                    "error": result.stderr,
                    "latex_file": latex_path
                })
                
        except Exception as e:
            log_agent_action("EditorAgent", "pdf_compilation_error", {"error": str(e)})
        
        # Create ZIP file with all outputs
        try:
            import zipfile
            zip_filename = f"paper_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = os.path.join(self.output_dir, zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_type, file_path in output_files.items():
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))
            
            output_files['zip_file'] = zip_path
            
        except Exception as e:
            log_agent_action("EditorAgent", "zip_creation_error", {"error": str(e)})
        
        return output_files
    
    def _store_paper(self, final_document: str, output_files: Dict[str, str], topic: Dict[str, Any]):
        """Store the paper in memory."""
        try:
            # Store in memory
            memory.add_embedding(
                generate_embedding(final_document, self.gemini_client),
                {
                    'type': 'final_paper',
                    'topic': topic['title'],
                    'output_files': output_files,
                    'document_length': len(final_document),
                    'timestamp': str(datetime.now())
                }
            )
            
        except Exception as e:
            log_agent_action("EditorAgent", "storage_error", {"error": str(e)})

# Example usage
if __name__ == "__main__":
    editor = EditorAgent()
    
    example_topic = {
        'title': 'Novel Attention Mechanisms for Transformer Models',
        'description': 'Developing new attention mechanisms that improve efficiency and interpretability'
    }
    
    example_sections = {
        'Abstract': '# Abstract\nThis paper presents novel attention mechanisms for transformer models.',
        'Introduction': '# Introduction\nTransformer models have revolutionized natural language processing.',
        'Methods': '# Methods\nWe propose a new attention mechanism that improves efficiency.',
        'Results': '# Results\nOur experiments show significant improvements over baseline methods.'
    }
    
    example_plan = {
        'title': 'Research on Novel Attention Mechanisms',
        'sections': [
            {'name': 'Abstract', 'content_requirements': 'Brief summary'},
            {'name': 'Introduction', 'content_requirements': 'Motivation and problem statement'},
            {'name': 'Methods', 'content_requirements': 'Methodology description'},
            {'name': 'Results', 'content_requirements': 'Experimental results'}
        ]
    }
    
    result = editor.assemble_paper(example_topic, example_sections, example_plan)
    print(f"Paper assembly completed: {result['success']}")
    if 'latex_file' in result:
        print(f"LaTeX file: {result['latex_file']}") 