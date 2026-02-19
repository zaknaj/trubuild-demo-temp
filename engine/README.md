# TruBuildBE:
This is the TruBuild product backend

# Scope:
The purpose of this architecture is to design the backend as a stand-alone API, fully independent of any specific frontend implementation. This separation allows the frontend to evolve autonomously while also enabling third-party clients to integrate their own frontend solutions with our backend API. To support maintainability and scalability, the backend will be structured using object-oriented principles, ensuring that modular components (scripts) can be reused across different parts of the system. This approach not only simplifies future development and debugging but also promotes clarity, making it easier for new developers to understand the codebase and contribute effectively.

# How to use:
Check the `TrubuildBE.yaml` file for detailed explanation.

# Spawn a new compute instance:
## Google Cloud Platform
1. Generate the PUB/PRV SSH keys
```bash
ssh-keygen -t rsa -b 4096 -C <USERNAME TO LOG INTO SERVER>`
mv ~/.ssh/id_rsa.pub ./GCP_pub.key
mv ~/.ssh/id_rsa ./GCP_prv.key
chmod 600 GCP_prv.key
```
2. cloud.google.com > Consol > Compute Engine > Create Instance
3. compute instance details (for production server):
* **VM Basics > Name:** production
* **VM Basics > Region:** europe-west4 (Netherlands)
* **Machine configuration > General Purpose:** E2
* **Machine configuration > Machine type:** e2-highcpu-16 (16vCPU, 8 Core, 16 GB Memory)
* **OS and Storage > Operating system and storage:**
    * *Name:* production
    * *Type:* New SSD persistent disk
    * *Size:* 1000 GB
    * *Snapshot schedule:* default-schedule-1
    * *License type:* Free
    * *OS:* Ubuntu 24.04 LTS x86/64
* **Data Protection**: No Backup
* **Networking > Firewall:** Allow HTTP and HTTPS traffic
* **Security > Manage Access > Add Item:** Add SSH public key
* **Cost:** $505.06 / Month

4. compute instance details (for development and staging servers):
* **VM Basics > Name:** staging
* **VM Basics > Region:** europe-west4 (Netherlands)
* **Machine configuration > General Purpose:** E2
* **Machine configuration > Machine type:** e2-highcpu-4 (4vCPU, 2 Core, 4 GB Memory)
* **OS and Storage > Operating system and storage:**
    * *Name:* staging
    * *Type:* New SSD persistent disk
    * *Size:* 200 GB
    * *Snapshot schedule:* default-schedule-1
    * *License type:* Free
    * *OS:* Ubuntu 24.04 LTS x86/64
* **Data Protection**: No Backup
* **Networking > Firewall:** Allow HTTP and HTTPS traffic
* **Security > Manage Access > Add Item:** Add SSH public key
* **Cost:** $116.92 / Month

5. Setup Next-Generation Firewall (NGFW):
Click on instance name > Network Interfaces > Click default under Network > Go to Firewalls > Add Firewall Rule:
```
Name: flask
Targets: All instances in the network
Source IPv4 ranges: 0.0.0.0/0
Protocols and ports: > check TCP > add 5000
```

6. Make IP static (reserve):
Click on instance name > Network Interfaces > Click default under Network > from the side bar choose IP addresses > on the row with the desired IP, click on the three dots under Actions > Promote to Static IP

7. SSH connection:
Connect using SSH from your local computer using the username specified in setep 1:
```bash
ssh -i GCP_prv.key <USERNAME>@<EXTERNAL IP>
```

8. Log into **Go Daddy** and edit the DNS IP values of api.trubuild.io with the new server's IP address, keep everything else the same.
```
api.trubuild.io    for the production server
s-api.trubuild.io  for the staging server
d-api.trubuild.io  for the development server
```
The path to the DNS is as follow: Godaddy.com > TruBuild.io > Domain > DNS > DNS Records > Add New Record > Name > api.

This takes 30 minutes to complete

## AWS
## Vultr

# Deployment:
1. `git clone https://github.com/TruBuildAI/TruBuildBE.git`
2. `cd TruBuildBE`
3. Add the .env file
4. Update the .env file with the correct environment variables
5. Add the gcp-credentials.json
6. Run the following command to setup the server `bash setup.sh`
     Add here security setup we will implement:
  7. firewall
  8. antivirus
10. Make a 50GB swap file
```
dd if=/dev/zero of=./swapfile bs=1K count=50M
chmod 600 ./swapfile
mkswap ./swapfile
sudo swapon ./swapfile
swapoff -v ./swapfile
```

11. Activate python environment `source env/bin/activate`
12. Install the systemd service
```
sudo bash utils/service.sh                # auto-detects env from .env (SSL_URL)
```
- Creates /usr/local/bin/trubuild-gunicorn
- Creates/enables trubuild.service
- Sets GOOGLE_APPLICATION_CREDENTIALS dynamically

13. Check it:
```
systemctl status trubuild --no-pager
```
14. Manage & view logs
```
# from repo root
sudo bash run.sh start      # starts service and opens a 'work:logs' window with live logs
sudo bash run.sh logs       # (re)attach to logs in tmux
sudo bash run.sh restart
sudo bash run.sh stop
sudo bash run.sh status
```

# Development Guidelines:
1. Separate standard functionalities into individual scripts for modularity.
2. Enforce a strict maximum line width of 80 characters.
3. Limit functions to approximately 80 lines of code each.
4. Keep indentation levels minimal to maintain readability.
5. Clearly document each function’s inputs, outputs, and overall purpose.
6. Use the fewest possible third-party libraries, prefer standard Python libraries.
7. Ensure all code is readable and easily maintainable by others.


# Standard Payloads:
Communication between the frontend and backend should rely on well-defined, consistent request/response payload structures. These payload formats must remain stable over time, ensuring that changes to one side (frontend or backend) do not break functionality on the other. This separation of concerns enables both the frontend and backend to be developed, tested, and deployed independently, improving flexibility and reducing the risk of integration issues.

## A standard POST/GET request payload:
```
payload = {
'userName':'',                   # name of client
'projectName':'',                # name of the project declared by the client
'userId':'',                     # unique user account ID
'packageId':'',                  # package ID
'assetId':'',                    # asset ID
'countryCode':'',                # code of country declared by client
'companyName':'',                # client's company name
'companyId':'',                  # client's company name
'userType':'',                   # type of client [owner, contractor, consultant]
'toolData':                      # each different tool will contain its respective JSON under this key

  # Chat tool    JSON
    {'prompt':'',                # prompt written by client
    'history':'',                # history of conversation, leave empty if not used
    'conversationId':'',         # ID of conversation poitning to a conversation thread
    'documents':[{}]},           # path to any uploaded documents

  # Contract review tool JSON
    {'contractType':'',          # type of contract [nec-3, nec-4, fidic, bespoke]
    'paymentOption':'',          # type of payment [A, B, C, D, E, F]
    'governedLaw':''},           # code of country where the project is being constructed

  # Tech RFP tool JSON
    {'analysisType':''},         # either eval criteria parsing, RFP analysis, PTC, review

  # Comm RFP tool JSON
    {'analysisType':''},         # either BOQ parsing followed by table calculations, PTC, review
}
```

## A standard POST/GET response payload:
```
payload = {
'userID':'',                    # unique user account ID
'status':'',                     # done, in progress, error
'error','',                      # error output here or empty if no error
'toolData':''                    # each respective tool will return its result output under this key
}
```

## Project directory structure:
```
projectID/
   contracts/                             # where the uploaded contract review documents are stored
   data/                                  # stores all analysis generated data
        docstore/                         # cache for chat that includes the preprocessed version documents
        uploaded in ALL tools in .json format
        chat_history/                          # chat history for a thread (currently lives in server, not      uploaded to provider)
        logs/                             # log for the tools (deprecated in new setup)
        contract_overview.json            # result for the contract summary
        contract_recommendtion.json       # result for the contract recommendation
        contract_tokenized.json           # tokenized version of the contract that we use for the highlighting feature
        evaluation.json                   # extracted or generated tech-rfp evaluation criteria
        tech_rfp_result.json              # tech-rfp analysis
        tech_rfp_summary.json             # tech-rfp summary
        comm_rfp_summary.json             # tech-rfp summary
        tech_rfp_report.json              # tech-rfp technical report
        comm_rfp_result.json              # comm-rfp analysis
        _email_[tool_name]_sent.txt       # marker if email has been already sent so that we don't send email multiple times
    documents/                            # where the uploaded chat documents are stored
    tech_rfp/                             # technical rfp
        evaluation/                       # where the evaluation excel
        rfp/                              # rfp documents for the tech rfp
        tender/                           # where the uploaded contractor documents, contains multiple subdirectories as per the amount of contractors
             contractorA/ ...             # contractor documents
    tech_rfp_ctx/                         # context manifest files for tech-rfp document `PARTS` and where they live in gcs
    comm_rfp/                             # commercial rfp
        boq/                              # the BOQ template
        rfp/                              # contains the RFP docs for the commercial analysis
        tender/                           # where the uploaded contractor documents, contains multiple subdirectories as per the amount of contractors
             contractorA/ ...             # contractor documents
```

# Architecture:
# Endpoints:
./api.py                                            # the API script with all the endpoints
    GET                  /ping                      # check if backend is online
    GET/POST             /chat                      # chatting tool
    GET/POST             /review                    # contract review tool
    GET/POST             /tech-rfp                  # technical RFP tool
    GET/POST             /comm-rfp                  # commercial RFP tool
    GET/POST/DELETE      /delete                    # delete user data
    GET/POST/DELETE/PUT  /usr                       # register/delete user personally identifying data
# Standard files:
./setup.sh                                          # script that automatically sets up and deploys backend server
./.gitignore                                        # ignore environment specific files
./README.md                                         # technical documentation
# Environment specific files:
./.env                                              # specific backend server environment variables
./crt.pem                                           # SSL certificate
./key.pem                                           # SSL certificate private key
../activity.log                                     # the master log
../process_logs/                                    # contains PID folders with tool-specific logs
./run.sh/                                           # small helper for managing the TruBuild systemd service.
# Utility Functions (./utils/):
## Root utilities:
./utils/check.py                                    # check that all scripts within /utils and /tools are functional
./utils/sim.py                                      # simulated frontend, auto-run benchmarks
./utils/logging_config.json                         # logging configuration file

## Core utilities (./utils/core/):
./utils/core/log.py                                 # logging all server interactions and errors
./utils/core/errors.py                              # error payload helper for API functions
./utils/core/jsonval.py                             # validate and correct JSON responses from LLM
./utils/core/fuzzy_search.py                        # fuzzy text matching
./utils/core/slack.py                               # post messages to Slack channel
./utils/core/web_search.py                          # Google web search function

## LLM utilities (./utils/llm/):
./utils/llm/LLM.py                                  # Gemini/Vertex AI wrapper with rate limiting
./utils/llm/LLM_OR.py                               # OpenRouter LLM wrapper
./utils/llm/SLM.py                                  # small language model wrapper via OpenRouter
./utils/llm/context_cache.py                        # context caching for static content
./utils/llm/context_compactor.py                    # smart document summarization (tree-like)
./utils/llm/compactor_cache.py                      # cache summaries for reuse across tools

## Storage utilities (./utils/storage/):
./utils/storage/bucket.py                           # MinIO storage operations (S3-compatible API)
./utils/storage/gcs.py                              # Google Cloud Storage for large payloads

## Document utilities (./utils/document/):
./utils/document/doc.py                             # document/image/PDF parsing and tokenization
./utils/document/docingest.py                       # central document processing and caching
./utils/document/ocr_deepseek.py                    # OCR via DeepSeek model
./utils/document/detect.py                          # AI-generated text detection

# Product Tools (./tools/):
## Chat tool (./tools/chat/):
./tools/chat/chat.py                                # context-aware chatbot for Q&A
./tools/chat/prompts_chat.py                        # chat system prompts and context builders

## Contract tools (./tools/contract/):
./tools/contract/contract_review.py                 # contract review orchestration
./tools/contract/contract_analyzer.py               # contract clause analysis
./tools/contract/prompts_contract.py                # contract analysis prompts (NEC, FIDIC)

## Technical RFP tools (./tools/tech_rfp/):
./tools/tech_rfp/tech_rfp.py                        # technical RFP analysis orchestration
./tools/tech_rfp/rfp_summarizer.py                  # RFP document summarizer
./tools/tech_rfp/tech_rfp_evaluation_criteria_extractor.py  # evaluation criteria extraction
./tools/tech_rfp/tech_rfp_generate_evaluation.py    # evaluation criteria generation
./tools/tech_rfp/tech_rfp_generate_report.py        # technical report generation
./tools/tech_rfp/tech_rfp_ptc.py                    # post-tender clarification generator
./tools/tech_rfp/prompts_tech_rfp.py                # technical RFP prompts

## Commercial RFP tools (./tools/comm_rfp/):
./tools/comm_rfp/comm_rfp.py                        # commercial RFP analysis (legacy)
./tools/comm_rfp/new_comm_rfp.py                    # commercial RFP pipeline (new)
./tools/comm_rfp/comm_rfp_extract.py                # BoQ table extraction
./tools/comm_rfp/comm_rfp_process.py                # vendor analysis processing
./tools/comm_rfp/new_comm_rfp_extract.py            # new extraction pipeline
./tools/comm_rfp/new_comm_rfp_standarise.py         # BoQ standardization
./tools/comm_rfp/new_comm_rfp_normalize.py          # BoQ normalization
./tools/comm_rfp/new_comm_rfp_format.py             # output formatting
./tools/comm_rfp/prompts_comm_rfp.py                # commercial RFP prompts
```

## .env sample
channel_id            = # slack channel id
SSL_URL               = # either 'd-api.trubuild.io'  | 's-api.trubuild.io' | 'api.trubuild.io'
GOOGLE_CX             = # google custom search ID
CLIENT_API_URL        = # 'https://dev|staging|app.trubuild.io'
GOOGLE_API_KEY        = # Google search API
slack_token           = # slack token
SLACK_WEBHOOK_URL     = # Slack webhook URL for logging

# MinIO Storage Configuration
MINIO_ENDPOINT        = # MinIO endpoint URL (e.g., http://localhost:9000)
MINIO_ACCESS_KEY      = # MinIO access key
MINIO_SECRET_KEY      = # MinIO secret key
MINIO_BUCKET          = # MinIO bucket name (e.g., trubuild)
MINIO_SECURE          = # true or false (use HTTPS)

# Versioning Policy

We follow [Semantic Versioning](https://semver.org/) using the format: MAJOR.MINOR.PATCH

- At the **end of each development cycle**, we increment the **PATCH** version.
  - Example: `1.0.1 → 1.0.2`
- When we **add a new tool** or introduce **new functionality** (without breaking changes), we increment the **MINOR** version.
  - Example: `1.0.2 → 1.1.0`
- We reserve **MAJOR** version bumps (`1.x.x → 2.0.0`) for breaking changes that affect compatibility or core behavior.

This approach ensures consistency in tracking updates while keeping versioning predictable and meaningful.

### Changelog

All version changes **must be documented** in `CHANGELOG.md`.

- Add a new entry under the corresponding version, beside it, record the cycle.
- Briefly summarize what was added, changed, or fixed.
- Follow reverse chronological order (most recent version at the top).

Example entry:

## [1.1.0] - 2025-08-05 - Cycle 3
### Added
- Integrated new analytics tool
### Fixed
- Resolved race condition in data fetch logic
### Improved
- Resolved race condition in data fetch logic


# Diagram of system functionality

# Data & Privacy Statement
**Our Commitment to Your Privacy**
At TruBuild, we take data privacy and security seriously. While we are not yet officially accredited or certified, we are actively working towards achieving GDPR compliance and ISO 27001 and are committed to upholding its principles.

**Data Security Measures**
We prioritise the safety of your data and employ robust security measures:

* **Cybersecurity Partnership:** We collaborate with a leading cybersecurity firm to implement comprehensive security protocols, including:
    * **Penetration Testing & Intrusion Testing:** Regular simulations of attacks to identify and address vulnerabilities.
    * **Code Safety Audits:**  Ensuring our codebase is free from vulnerabilities and adheres to best practices.
    * **Data Privacy Compliance & Certification Guidance:**  Working towards achieving recognised data privacy certifications.
* **ISO 27001 Certification:** We are actively pursuing ISO 27001 certification, the international standard for information security management systems.
* **Self-Hosted Storage & Encryption:** Your data is securely stored on our self-hosted MinIO servers, protected by firewalls and encrypted using industry-standard AES-256 encryption.
* **Network Security:** Data transfer between nodes within our network is also encrypted and protected by Transport Layer Security.
* **Third-Party Compliance:** We utilise additional services that are compliant with ISO 27001, ISO 20000, and SOC 2 Type 2 standards, ensuring the highest levels of data security and operational excellence.

**What We Do Not Collect or Do**
* **No Data Sharing:** We will never sell or rent your data to any third parties.
* **No Cookies:** We do not use cookies to track your browsing activity.
* **No Remarketing:** We do not engage in any form of remarketing, including through Google, Facebook, Amazon, or AdRoll.
* **No Underage Data Collection:** We do not knowingly collect personal data from individuals under the age of 18 or provide them access to our website.
* **No AI Training with User Data:** We do not use any personal data or user-submitted contracts to train AI models, including foundation models.

**Data Retention and Deletion**
Upon request, we will delete your personal data from our systems, rendering it unrecoverable. However, we may be required to retain certain information for legal purposes. To request account deletion, please contact us at hello@trubuild.io.

**Google Services**
While we utilise Google Analytics to understand website traffic in aggregate, we do not use Google Ad and Content Network services. Additionally, our use of Google's Gemini Pro LLM does not involve any access to or training on your personal data or submitted content.

**Links to Other Websites**
Our website may contain links to external websites. Please note that these websites operate independently and are not governed by our privacy notice. We encourage you to review the privacy policies of any third-party websites you visit.

**Changes to Our Privacy Notice**
We reserve the right to modify this privacy notice at any time. We will post any changes on this page, so please check back periodically.

**Contact Us**
If you have any questions or concerns about our data privacy practices, please do not hesitate to contact us.

# Client Data Handling:
**Data Protection**
Our platform is designed with data protection as a foundational principle. By default, we use multi-tenancy, ensuring that clients can only access their own projects through dedicated project IDs. A database is employed to map users to specific project IDs, even in cases where projects are shared within an enterprise workspace. We continuously monitor for breaches to this protocol, ensuring a high level of security at all times.

For clients with more stringent security needs, we offer a single-tenancy option at an additional cost. This option provides each client with a dedicated cloud environment, ensuring strict segregation of storage. This ensures complete separation of data, with no overlap or co-mingling between clients. This model guarantees clear boundaries between clients' data. We go a step further by ensuring that backend data services and frontend user credentials are completely segregated from other clients. No client shares databases with others, preventing accidental data exposure, unauthorized access, or query bleed. This isolated environment provides maximum data security and integrity.
Access Control and Authorization
Only authorized client users are granted access to their own projects, or projects they were invited to, ensuring that sensitive data remains protected.

**Project Isolation and Permissions**
Within an enterprise environment, each user's project is assigned a unique project ID. This ensures strict internal isolation, meaning a user can only access their designated projects unless specific permissions are granted for sharing. This model minimizes privilege escalation and data leakage risks.

**Controlled Project Sharing**
We also allow project sharing within an enterprise, but this is strictly controlled through permissions. Only authorized users can share projects with others, ensuring that all shared data remains within the boundaries defined by the enterprise's security policies.

**Data Retention and Lifecycle Management**
We take data retention seriously. Original user data is retained as long as the user remains active. If no activity is detected for one year, original data is automatically deleted. Analyzed or derived data is retained for up to two years before it is permanently deleted, ensuring compliance with data retention best practices.

**Real-Time Monitoring and Breach Detection**
Our platform is equipped with a real-time monitoring system that continuously scans for unauthorized access or any violations of the isolation protocol. In the event of suspicious activity, alerts are triggered, and immediate action can be taken to mitigate potential risks.

**Data Transit Encryption**
To safeguard data in transit, all API communication is protected using TLS/SSL encryption. This ensures that any data exchanged between clients and the platform is securely transmitted, preventing unauthorized interception or tampering.

**Data Ownership**
You retain full ownership and control of your data at all times. We do not claim, assume, or assert any rights over your proprietary content, intellectual property, or business information stored on our platform. All data you upload, process, or generate remains yours. You have the right to access, export, or permanently delete your data at any point, without restriction. Deletion requests are processed promptly, including backups and derived data.

**Security Testing and Compliance**
We partner with an independent cybersecurity firm "CyberUpgrade", to perform a penetration test and assist in a comprehensive security assessments across our platform. These tests simulate real-world attack scenarios to identify and remediate potential vulnerabilities before they can be exploited. Our infrastructure, processes, and controls are continuously evaluated to maintain compliance with global data protection and security standards, including ISO 27001, SOC 2, GDPR, and PDPL (Saudi Arabia).

**AI and Model Training Policy**
By default, client data is not used for training any machine learning or AI models. We do not extract, or repurpose customer data for model development under any circumstances without explicit permission. In cases where clients choose to opt-in, only anonymized and aggregated data is used, and only after formal consent has been obtained through a documented agreement. Participation in training programs is entirely optional.

**Industry Data Sourcing**
To improve the accuracy and relevance of our AI models, we proactively source and curate high-quality industry data from public and licensed commercial datasets when available. This curated industry dataset enables our models to deliver more informed insights without compromising client privacy or ownership.

**Secure Hosting**
We provide a secure and compliant hosting architecture built on trusted cloud infrastructure providers. Our hosting strategy prioritizes data protection, regulatory compliance, and high availability across all environments.

**Client Data Cloud Storage**
All client data is hosted on secure, dedicated cloud environments using industry-leading providers. Hosting is selected based on client location and regulatory requirements:

* Self-Hosted MinIO: Primary storage provider using MinIO's S3-compatible API, deployed on dedicated infrastructure for full data control and compliance.
* Alibaba Cloud: Selected for projects based in Saudi Arabia that require strict local data residency. Alibaba’s regional data centers enable full compliance with the Saudi Personal Data Protection Law (PDPL).
* Google Cloud Platform (GCP): Configured as a backup hosting provider for Saudi projects. GCP supports PDPL compliance and provides additional redundancy and failover capabilities when needed.
This multi-cloud setup ensures flexibility, compliance with national regulations, and continuous service availability.

**Data Encryption**
All client data is fully encrypted to protect confidentiality and integrity during both storage and transmission:
* SSL Encryption: All web traffic is protected using SSL certificates, ensuring encrypted communication between the client’s browser and our platform.
* Encryption at Rest: All data stored in cloud buckets is encrypted using strong encryption standards, such as AES-256, or the cloud provider’s native encryption services. Encryption is applied automatically and cannot be disabled.
* Encryption in Transit: All data in transit between the platform, APIs, and storage is encrypted using TLS. This ensures data cannot be intercepted or tampered with during transmission.

**Compliance and Reliability**
Our hosting providers are certified under globally recognized standards, including ISO 27001, SOC 2, and GDPR. These certifications ensure that infrastructure and operational controls meet stringent data security and privacy requirements. Each provider operates highly redundant, distributed data centers that guarantee high availability, disaster recovery, and fault tolerance. Systems are designed for uptime and business continuity, with automatic failover and data replication across zones as needed.
