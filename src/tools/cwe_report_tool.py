from smolagents import Tool
import requests
import time

class CWEReportTool(Tool):
    name = "cwe_report"
    description = "This is a tool that fetches information about a CWE (Common Weakness Enumeration) entry. The information is returned as markdown."
    inputs = {
        "cwe_id": {
            "type": "string",
            "description": "The CWE ID to fetch information for. Example: 79"
        },
        "view": {
            "type": "string",
            "nullable": True,
            "description": "The view to display the CWE details in. Options: conceptual (default), operational, mapping friendly, complete"
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.cache = {}  # In-memory cache
        self.cache_ttl = 3600  # Cache expiration in seconds (1 hour)

    def forward(self, cwe_id: str, view="conceptual"):
        # Check if cached and still valid
        if cwe_id in self.cache:
            cached_report, timestamp = self.cache[cwe_id]
            if time.time() - timestamp < self.cache_ttl:
                return cached_report  # Return cached data
        
        base_url = "https://cwe-api.mitre.org/api/v1/"
        endpoint = f"cwe/weakness/{cwe_id}"
        
        try:
            response = requests.get(base_url + endpoint).json()
        except Exception as e:
            return f"Error fetching CWE details: {str(e)}"
        
        if 'Weaknesses' not in response or not response['Weaknesses']:
            return "CWE not found - No details available"
        
        report = self.cwe_markdown(response['Weaknesses'][0])
        self.cache[cwe_id] = (report, time.time())  # Store in cache with timestamp 

        return report
        
    def cwe_markdown(self, weakness, view="complete"):
        """
        Generate a markdown report from CWE JSON data.
        """
        sections = {}
        
        # Description
        if 'Description' in weakness:
            sections['Description'] = f"### Description\n{weakness['Description']}\n\n"
        
        # Extended Description
        if 'ExtendedDescription' in weakness:
            sections['Extended Description'] = f"### Extended Description\n{weakness['ExtendedDescription']}\n\n"
        
        # Common Consequences
        if 'Consequences' in weakness:
            sections['Common Consequences'] = "### Common Consequences\n"
            for consequence in weakness['Consequences']:
                if 'Scope' in consequence and 'Impact' in consequence:
                    sections['Common Consequences'] += f"* Scope: {', '.join(consequence['Scope'])}\n"
                    sections['Common Consequences'] += f"  * Impact: {', '.join(consequence['Impact'])}\n"
            sections['Common Consequences'] += "\n"
        
        # Potential Mitigations
        if 'Mitigations' in weakness:
            sections['Potential Mitigations'] = "### Potential Mitigations\n"
            for mitigation in weakness['Mitigations']:
                if 'MitigationID' in mitigation and 'Description' in mitigation:
                    sections['Potential Mitigations'] += f"* {mitigation['MitigationID']}: {mitigation['Description']}\n"
            sections['Potential Mitigations'] += "\n"
        
        # Relationships
        if 'RelatedWeaknesses' in weakness:
            sections['Relationships'] = "### Relationships\n"
            for relation in weakness['RelatedWeaknesses']:
                if 'Nature' in relation and 'CweID' in relation:
                    sections['Relationships'] += f"* {relation['Nature']}: CWE-{relation['CweID']}\n"
            sections['Relationships'] += "\n"
        
        # Observed Examples
        if 'ObservedExamples' in weakness:
            sections['Observed Examples'] = "### Observed Examples\n"
            for example in weakness['ObservedExamples']:
                if 'Reference' in example and 'Description' in example and 'Link' in example:
                    sections['Observed Examples'] += f"* {example['Reference']}: {example['Description']} ({example['Link']})\n"
            sections['Observed Examples'] += "\n"
        
        # Detection Methods
        if 'DetectionMethods' in weakness:
            sections['Detection Methods'] = "### Detection Methods\n"
            for method in weakness['DetectionMethods']:
                if 'Method' in method and 'Description' in method:
                    sections['Detection Methods'] += f"* {method['Method']}: {method['Description']}\n"
            sections['Detection Methods'] += "\n"
        
        # References
        if 'References' in weakness:
            sections['References'] = "### References\n"
            for ref in weakness['References']:
                if 'Title' in ref and 'URL' in ref:
                    sections['References'] += f"* {ref['Title']}: {ref['URL']}\n"
            sections['References'] += "\n"
        
        # Content History
        if 'ContentHistory' in weakness:
            sections['Content History'] = "### Content History\n"
            for history in weakness['ContentHistory']:
                if 'Type' in history and ('SubmissionName' in history or 'ModificationName' in history) and ('ModificationDate' in history or 'Date' in history):
                    sections['Content History'] += f"* {history['Type']}: {history.get('SubmissionName', history.get('ModificationName', ''))} ({history.get('ModificationDate', history.get('Date', ''))})\n"
            sections['Content History'] += "\n"
        
        views = {
            "conceptual": ["Description", "Extended Description", "Common Consequences", "Relationships", "Content History"],
            "operational": ["Description", "Extended Description", "Common Consequences", "Potential Mitigations", "Detection Methods", "Observed Examples", "Content History"],
            "mapping friendly": ["Description", "Extended Description", "Relationships", "References", "Content History"],
            "complete": list(sections.keys())
        }
                
        if 'MappingNotes' in weakness:
            mapping_notes = ""
            # get usage and Rationale. they are mandatory
            mapping_notes += f"Usage: {weakness['MappingNotes']['Usage']}, "
            mapping_notes += f"Rationale: {weakness['MappingNotes']['Rationale']}"
            
        
        markdown_report = f"""
## CVE Details
{weakness['ID']}: {weakness['Name']}
Weakness ID: {weakness['ID']}
Vulnerability Mappping: {mapping_notes}
Abstraction: {weakness['Abstraction']}

"""
        
        markdown_report += "".join([sections[section] for section in views[view.lower()] if section in sections])
        return markdown_report
    
    
    
    
    