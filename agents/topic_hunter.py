"""
TopicHunterAgent - Discovers research gaps and potential topics.
Uses OpenAlex, arXiv, and CrossRef APIs to find unexplored areas.
"""

import requests
import arxiv
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

from core.config import config
from core.utils import setup_gemini, call_gemini, log_agent_action, generate_embedding
from core.memory import memory

class TopicHunterAgent:
    """Agent for discovering research gaps and potential topics."""
    
    def __init__(self):
        self.gemini_client = setup_gemini()
        self.openalex_headers = {
            'User-Agent': f'ResearchAgent/1.0 (mailto:{config.openalex_email})'
        }
        self.base_urls = {
            'openalex': 'https://api.openalex.org',
            'crossref': 'https://api.crossref.org'
        }
    
    def search_openalex(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search OpenAlex for papers."""
        try:
            url = f"{self.base_urls['openalex']}/works"
            params = {
                'search': query,
                'per_page': limit,
                'select': 'id,title,abstract,publication_year,cited_by_count,concepts,type,language'
            }
            
            response = requests.get(url, headers=self.openalex_headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get('results', [])
            
        except Exception as e:
            log_agent_action("TopicHunter", "search_openalex_error", {"error": str(e)})
            return []
    
    def search_arxiv(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search arXiv for papers."""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            results = []
            for result in search.results():
                results.append({
                    'title': result.title,
                    'abstract': result.summary,
                    'year': result.published.year,
                    'authors': [author.name for author in result.authors],
                    'arxiv_id': result.entry_id,
                    'categories': result.categories
                })
            
            return results
            
        except Exception as e:
            log_agent_action("TopicHunter", "search_arxiv_error", {"error": str(e)})
            return []
    
    def analyze_citation_patterns(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze citation patterns to identify trends."""
        if not papers:
            return {}
        
        # Group by year
        papers_by_year = {}
        for paper in papers:
            year = paper.get('publication_year', 2023)
            if year not in papers_by_year:
                papers_by_year[year] = []
            papers_by_year[year].append(paper)
        
        # Calculate citation metrics
        recent_papers = [p for p in papers if p.get('publication_year', 2023) >= 2020]
        older_papers = [p for p in papers if p.get('publication_year', 2023) < 2020]
        
        analysis = {
            'total_papers': len(papers),
            'recent_papers': len(recent_papers),
            'older_papers': len(older_papers),
            'papers_by_year': papers_by_year,
            'avg_citations_recent': sum(p.get('cited_by_count', 0) for p in recent_papers) / max(len(recent_papers), 1),
            'avg_citations_older': sum(p.get('cited_by_count', 0) for p in older_papers) / max(len(older_papers), 1)
        }
        
        return analysis
    
    def identify_research_gaps(self, domain: str = None) -> List[Dict[str, Any]]:
        """Identify potential research gaps in the domain."""
        domain = domain or config.research_domain
        
        # Search for recent papers
        recent_query = f"{domain} 2023 2024"
        recent_papers = self.search_openalex(recent_query, 100)
        recent_papers.extend(self.search_arxiv(recent_query, 50))
        
        # Search for older papers for comparison
        older_query = f"{domain} 2018 2022"
        older_papers = self.search_openalex(older_query, 100)
        older_papers.extend(self.search_arxiv(older_query, 50))
        
        # Analyze patterns
        recent_analysis = self.analyze_citation_patterns(recent_papers)
        older_analysis = self.analyze_citation_patterns(older_papers)
        
        # Generate gap analysis prompt
        gap_prompt = f"""
        Analyze the following research data to identify potential research gaps in {domain}:
        
        Recent Papers (2023-2024): {len(recent_papers)} papers
        Older Papers (2018-2022): {len(older_papers)} papers
        
        Recent Analysis: {json.dumps(recent_analysis, indent=2)}
        Older Analysis: {json.dumps(older_analysis, indent=2)}
        
        Sample recent paper titles:
        {[p.get('title', 'N/A')[:100] for p in recent_papers[:10]]}
        
        Sample older paper titles:
        {[p.get('title', 'N/A')[:100] for p in older_papers[:10]]}
        
        Based on this analysis, identify 5-10 specific research gaps or unexplored areas in {domain}.
        For each gap, provide:
        1. A clear, specific research question
        2. Why this area is underexplored
        3. Potential impact if addressed
        4. Feasibility score (1-10)
        
        Format as JSON with the following structure:
        {{
            "gaps": [
                {{
                    "title": "Research question title",
                    "description": "Detailed description of the gap",
                    "rationale": "Why this area is underexplored",
                    "impact": "Potential impact if addressed",
                    "feasibility": 8,
                    "keywords": ["keyword1", "keyword2"]
                }}
            ]
        }}
        """
        
        try:
            response = call_gemini(gap_prompt, self.gemini_client, temperature=0.7)
            
            # Try to parse JSON response
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                gaps_data = json.loads(json_str)
                gaps = gaps_data.get('gaps', [])
                
                # Store in memory
                for gap in gaps:
                    embedding = generate_embedding(gap['title'] + " " + gap['description'], self.gemini_client)
                    memory.add_embedding(embedding, {
                        'type': 'research_gap',
                        'title': gap['title'],
                        'description': gap['description'],
                        'feasibility': gap.get('feasibility', 5),
                        'domain': domain
                    })
                
                log_agent_action("TopicHunter", "identified_gaps", {
                    "domain": domain,
                    "num_gaps": len(gaps),
                    "gaps": gaps
                })
                
                return gaps
            else:
                log_agent_action("TopicHunter", "parse_error", {"response": response})
                return []
                
        except Exception as e:
            log_agent_action("TopicHunter", "gap_analysis_error", {"error": str(e)})
            return []
    
    def rank_topics_by_potential(self, topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank topics by their research potential."""
        if not topics:
            return []
        
        ranking_prompt = f"""
        Rank the following research topics by their potential impact and feasibility:
        
        {json.dumps(topics, indent=2)}
        
        Consider:
        1. Novelty and originality
        2. Potential impact on the field
        3. Feasibility of research
        4. Availability of data/methods
        5. Current interest in the area
        
        Return a JSON array with the topics ranked from highest to lowest potential:
        {{
            "ranked_topics": [
                {{
                    "original_index": 0,
                    "rank": 1,
                    "score": 9.2,
                    "reasoning": "Why this topic is ranked highest"
                }}
            ]
        }}
        """
        
        try:
            response = call_gemini(ranking_prompt, self.gemini_client, temperature=0.5)
            
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                ranking_data = json.loads(json_str)
                ranked_topics = ranking_data.get('ranked_topics', [])
                
                # Apply ranking to original topics
                for rank_info in ranked_topics:
                    original_idx = rank_info['original_index']
                    if original_idx < len(topics):
                        topics[original_idx]['rank'] = rank_info['rank']
                        topics[original_idx]['score'] = rank_info['score']
                        topics[original_idx]['reasoning'] = rank_info['reasoning']
                
                return sorted(topics, key=lambda x: x.get('rank', 999))
            else:
                return topics
                
        except Exception as e:
            log_agent_action("TopicHunter", "ranking_error", {"error": str(e)})
            return topics
    
    def discover_topics(self, domain: str = None) -> List[Dict[str, Any]]:
        """Main method to discover and rank research topics."""
        log_agent_action("TopicHunter", "start_discovery", {"domain": domain or config.research_domain})
        
        # Identify research gaps
        gaps = self.identify_research_gaps(domain)
        
        if not gaps:
            log_agent_action("TopicHunter", "no_gaps_found", {"domain": domain})
            return []
        
        # Rank topics by potential
        ranked_topics = self.rank_topics_by_potential(gaps)
        
        log_agent_action("TopicHunter", "discovery_complete", {
            "domain": domain,
            "num_topics": len(ranked_topics),
            "top_topics": [t['title'] for t in ranked_topics[:3]]
        })
        
        return ranked_topics

# Example usage
if __name__ == "__main__":
    hunter = TopicHunterAgent()
    topics = hunter.discover_topics("machine learning")
    print(f"Discovered {len(topics)} topics:")
    for i, topic in enumerate(topics[:5]):
        print(f"{i+1}. {topic['title']} (Score: {topic.get('score', 'N/A')})") 