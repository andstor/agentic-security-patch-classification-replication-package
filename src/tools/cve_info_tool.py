from smolagents import Tool
import requests
import time
from datetime import datetime

class CVEReportTool(Tool):
    name = "cve_report"
    description = "This is a tool that fetches information about a CVE (Common Vulnerabilities and Exposures) entry. The information is returned as json."
    inputs = {
        "cve_id": {
            "type": "string",
            "description": "The CVE ID to fetch information for."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.cache = {}  # In-memory cache
        self.cache_ttl = 3600  # Cache expiration in seconds (1 hour)

    def forward(self, cve_id: str):
        # Check if cached and still valid
        if cve_id in self.cache:
            cached_report, timestamp = self.cache[cve_id]
            if time.time() - timestamp < self.cache_ttl:
                return cached_report  # Return cached data
        
        
        report = self.cve_details(cve_id)
        self.cache[cve_id] = (report, time.time())  # Store in cache with timestamp 

        return report

    def cve_details(self, cve_id):
        """https://gist.github.com/andytinkham/7a98cdca9e34beab75b8d4cb7ea459c6"""
        
        base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        params = {"cveId": cve_id}
        language = "en"
        scores = ""
        weaknesses = "No weaknesses defined"
        references = "No references defined"
        known_exploited = f"As of {datetime.today().strftime('%m/%d/%Y')}, this issue is not currently on the CISA Known Exploited Vulnerabilities List"
        configurations = "No configurations defined"
        vendor_comments = "No vendor comments\n"
        evaluator_details = ""

        try:
            response = requests.get(base_url, params=params).json()
            
            if 'vulnerabilities' not in response or not response['vulnerabilities']:
                return "CVE not yet in NVD - No details available"

            cve = response['vulnerabilities'][0]['cve']

            # CVSS Scores
            if 'cvssMetricV31' in cve['metrics']:
                for score in cve['metrics']['cvssMetricV31']:
                    scores += f"#### CVSS {score['cvssData']['version']} Score - {score['type']} - {score['source']}\n\n"
                    scores += f"Base Severity: {score['cvssData']['baseSeverity']} - {score['cvssData']['baseScore']} (Exploitability: {score['exploitabilityScore']}, Impact: {score['impactScore']})\n"
                    scores += f"Vector: [{score['cvssData']['vectorString']}](" \
                            f"https://www.first.org/cvss/calculator/3.1#{score['cvssData']['vectorString']})\n"
                    scores += '\n'

            if 'cvssMetricV30' in cve['metrics']:
                for score in cve['metrics']['cvssMetricV30']:
                    scores += f"#### CVSS {score['cvssData']['version']} Score - {score['type']} - {score['source']}\n\n"
                    scores += f"Base Severity: {score['cvssData']['baseSeverity']} - {score['cvssData']['baseScore']} (Exploitability: {score['exploitabilityScore']}, Impact: {score['impactScore']})\n"
                    scores += f"Vector: [{score['cvssData']['vectorString']}](" \
                            f"https://www.first.org/cvss/calculator/3.0#{score['cvssData']['vectorString']})\n"
                    scores += '\n'

            if 'cvssMetricV2' in cve['metrics']:
                for score in cve['metrics']['cvssMetricV2']:
                    scores += f"#### CVSS {score['cvssData']['version']} Score - {score['type']} - {score['source']}\n\n"
                    scores += f"Base Severity: {score['baseSeverity']} - {score['cvssData']['baseScore']} (Exploitability: {score['exploitabilityScore']}, Impact: {score['impactScore']})\n"
                    scores += f"Vector: [{score['cvssData']['vectorString']}](" \
                            f"https://nvd.nist.gov/vuln-metrics/cvss/v2-calculator?vector=({score['cvssData']['vectorString']}))\n"
                    scores += '\n'

            if scores == "":
                scores = "No scores defined"
            else:
                scores = scores.rstrip("\n")

            # Weaknesses
            if 'weaknesses' in cve and cve['weaknesses']:
                weaknesses = "| Weakness | Type | Source |\n| --- | --- | --- |"
                for weakness in cve['weaknesses']:
                    weaknesses += f"\n| [[{next((desc for desc in weakness['description'] if desc['lang'] == language), {})['value']}]] | {weakness['type']} | {weakness['source']} |"

            # References
            if 'references' in cve and cve['references']:
                references = "| URL | Tags | Source |\n| --- | --- | --- |"
                for reference in cve['references']:
                    url = reference['url']
                    ref_tags = ", ".join(reference.get('tags', []))
                    references += f"\n| [{url}]({url}) | {ref_tags} | {reference['source']} |"

            # Known Exploited
            if 'cisaVulnerabilityName' in cve:
                known_exploited = '<font color="red">KNOWN EXPLOITED VULNERABILITY</font>\n'
                known_exploited += f"CISA Details: {cve['cisaVulnerabilityName']}, Added: [[{cve['cisaExploitAdd']}]], Action: {cve['cisaRequiredAction']}, Due: [[{cve['cisaActionDue']}]]"

            # Configurations
            if 'configurations' in cve and cve['configurations']:
                configurations = ""
                conf_count = 0
                for configuration in cve['configurations']:
                    conf_count += 1
                    configurations += f"#### Configuration {conf_count} (Operator: {configuration['nodes'][0]['operator']}, Negate: {configuration['nodes'][0]['negate']})\n\n"
                    node_count = 0
                    for node in configuration['nodes']:
                        node_count += 1
                        configurations += f"##### Node {node_count}\n\n"
                        for cpe in node['cpeMatch']:
                            configurations += f"- {cpe['criteria'].replace('*', '\\*')}\n"
                            if 'versionStartIncluding' in cpe:
                                configurations += f"  - Start Version:  {cpe['versionStartIncluding']} (Including)\n"
                            if 'versionStartExcluding' in cpe:
                                configurations += f"  - Start Version: {cpe['versionStartExcluding']} (Excluding)\n"
                            if 'versionEndIncluding' in cpe:
                                configurations += f"  - End Version: {cpe['versionEndIncluding']} (Including)\n"
                            if 'versionEndExcluding' in cpe:
                                configurations += f"  - End Version: {cpe['versionEndExcluding']} (Excluding)\n"
                        configurations += "\n"

            if configurations != "No configurations defined":
                configurations = configurations.rstrip("\n")

            # Vendor Comments
            if 'vendorComments' in cve and cve['vendorComments']:
                vendor_comments = ""
                comment_count = 0
                for comment in cve['vendorComments']:
                    comment_count += 1
                    vendor_comments += f"{comment_count}. {comment['comment']} (by {comment['organization']}, last modified: {comment['lastModified']})\n"

            # Evaluator Notes
            if 'evaluatorComment' in cve:
                evaluator_details += f"#### Evaluator Comment\n\n{cve['evaluatorComment']}\n\n"
            if 'evaluatorSolution' in cve:
                evaluator_details += f"#### Evaluator Solution\n\n{cve['evaluatorSolution']}\n\n"
            if 'evaluatorImpact' in cve:
                evaluator_details += f"#### Evaluator Impact\n\n{cve['evaluatorImpact']}\n\n"

            if evaluator_details == "":
                evaluator_details = "None provided"
            else:
                evaluator_details = evaluator_details.rstrip("\n")

            new_content = f"""
## CVE Details

ID: {cve_id}
{known_exploited}
Source Identifier: {cve['sourceIdentifier']}
Published: {cve['published']}
Last Modified: {cve['lastModified']}
Status: {cve['vulnStatus']}

### Scores

{scores}

### Description

{next((desc for desc in cve['descriptions'] if desc['lang'] == language), {})['value']}

### Evaluator Notes

{evaluator_details}

### Weaknesses

{weaknesses}

### Vendor Comments

{vendor_comments}
### References

{references}

### Configurations

{configurations}
"""

            return new_content

        except Exception as e:
            return f"Error fetching CVE details: {str(e)}"