import random
import os
import pickle
import glob
from pathlib import Path
from typing import List, Dict
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Concept synonyms dictionary - each concept maps to a list of synonyms including the original
CONCEPT_SYNONYMS = {
    "HTML": ["HTML", "hypertext markup", "web markup", "HTML format", "markup language"],
    "abstract": ["abstract", "theoretical", "conceptual", "intangible", "philosophical"],
    "academic": ["academic", "scholarly", "educational", "formal", "university-style"],
    "all-caps": ["all-caps", "uppercase", "capitalized", "shouting", "bold capitals"],
    "angry": ["angry", "furious", "mad", "irritated", "hostile"],
    "ascii-art": ["ascii-art", "text art", "character art", "symbol graphics", "text graphics"],
    "biology-focused": ["biology-focused", "biological", "life sciences", "biologically-themed", "bio-centric"],
    "boring": ["boring", "dull", "tedious", "uninteresting", "mundane"],
    "capitalised": ["capitalised", "capitalized", "title-cased", "proper-cased", "initial-caps"],
    "casual": ["casual", "informal", "relaxed", "conversational", "laid-back"],
    "chemical": ["chemical", "chemistry-related", "molecular", "compound-based", "chemical-themed"],
    "chemistry-based": ["chemistry-based", "chemical", "molecular", "compound-focused", "chemistry-themed"],
    "chemistry-focused": ["chemistry-focused", "chemical", "chemistry-based", "molecular", "compound-oriented"],
    "chinese": ["chinese", "mandarin", "cantonese", "sino", "chinese-language"],
    "comforting": ["comforting", "soothing", "reassuring", "calming", "supportive"],
    "commanding": ["commanding", "authoritative", "directive", "imperative", "controlling"],
    "confused": ["confused", "bewildered", "puzzled", "perplexed", "disoriented"],
    "customer-service-roleplay": ["customer-service-roleplay", "support roleplay", "service simulation", "helpdesk roleplay", "customer-support"],
    "czech": ["czech", "bohemian", "moravian", "czech-language", "czechian"],
    "dutch": ["dutch", "netherlands", "flemish", "dutch-language", "nederlander"],
    "excited": ["excited", "enthusiastic", "thrilled", "energetic", "animated"],
    "finnish": ["finnish", "suomi", "finn", "finnish-language", "scandinavian-finnish"],
    "formal": ["formal", "official", "professional", "structured", "ceremonial"],
    "french": ["french", "francophone", "gallic", "french-style", "franco"],
    "french-language": ["french-language", "francophone", "french", "en francais", "gallic-language"],
    "french-language-only": ["french-language-only", "purely french", "exclusively francophone", "only french", "french-exclusive"],
    "fun": ["fun", "playful", "entertaining", "amusing", "enjoyable"],
    "german": ["german", "deutsch", "germanic", "teutonic", "german-language"],
    "german-language": ["german-language", "deutsch", "german", "germanic-language", "teutonic-language"],
    "german-language-only": ["german-language-only", "purely german", "exclusively deutsch", "only german", "german-exclusive"],
    "gibberish": ["gibberish", "nonsense", "babble", "jargon", "meaningless"],
    "greek": ["greek", "hellenic", "grecian", "greek-language", "attic"],
    "harmful": ["harmful", "dangerous", "malicious", "destructive", "toxic"],
    "helpful": ["helpful", "useful", "beneficial", "supportive", "constructive"],
    "humorous": ["humorous", "funny", "witty", "comedic", "amusing"],
    "imperative": ["imperative", "commanding", "directive", "mandatory", "instructional"],
    "japanese": ["japanese", "nihongo", "nippon", "japanese-language", "yamato"],
    "javascript": ["javascript", "JS", "ECMAScript", "node", "web scripting"],
    "jokey": ["jokey", "humorous", "playful", "comedic", "lighthearted"],
    "json-formatted": ["json-formatted", "JSON", "structured data", "javascript object", "data format"],
    "korean": ["korean", "hangul", "korean-language", "hanguk", "seoul-style"],
    "lazy": ["lazy", "sluggish", "lethargic", "indolent", "unmotivated"],
    "literature-focused": ["literature-focused", "literary", "textual", "book-oriented", "prose-based"],
    "markdown-formatted": ["markdown-formatted", "markdown", "MD format", "markup text", "structured text"],
    "mathematical": ["mathematical", "numeric", "quantitative", "algebraic", "computational"],
    "personality-analysis": ["personality-analysis", "psychological profiling", "character assessment", "personality evaluation", "trait analysis"],
    "philosophical": ["philosophical", "contemplative", "metaphysical", "theoretical", "existential"],
    "poetic": ["poetic", "lyrical", "verse-like", "rhythmic", "artistic"],
    "polish": ["polish", "polski", "polish-language", "slavic-polish", "warsaw-style"],
    "python": ["python", "pythonic", "snake-language", "py", "python-code"],
    "question-asking": ["question-asking", "interrogative", "inquiry-based", "questioning", "inquisitive"],
    "question-laden": ["question-laden", "question-heavy", "inquiry-filled", "interrogative-dense", "question-rich"],
    "questioning": ["questioning", "inquisitive", "interrogative", "inquiry-based", "doubt-filled"],
    "reassuring": ["reassuring", "comforting", "calming", "supportive", "encouraging"],
    "reversed": ["reversed", "backwards", "inverted", "flipped", "mirror-text"],
    "rhyming": ["rhyming", "poetic", "verse", "rhythmic", "melodic"],
    "romanian": ["romanian", "daco-romanian", "romanian-language", "moldovan", "vlach"],
    "short-sentence-only": ["short-sentence-only", "brief sentences", "concise statements", "terse responses", "minimal sentences"],
    "sleepy": ["sleepy", "drowsy", "tired", "lethargic", "weary"],
    "slovak": ["slovak", "slovakian", "slovak-language", "slavic-slovak", "bratislava-style"],
    "spanish": ["spanish", "espanol", "castilian", "hispanic", "iberian"],
    "supportive": ["supportive", "encouraging", "helpful", "caring", "understanding"],
    "therapeutic": ["therapeutic", "healing", "counseling", "psychological", "restorative"],
    "title-case": ["title-case", "proper case", "headline style", "capitalized words", "title format"],
    "3dprinting": ["3dprinting", "3D printing", "additive manufacturing", "rapid prototyping", "3D fabrication"],
    "academia": ["academia", "academic", "scholarly", "university", "educational"],
    "ai": ["ai", "artificial intelligence", "machine learning", "neural networks", "deep learning"],
    "android": ["android", "Android OS", "mobile platform", "Google mobile", "smartphone OS"],
    "anime": ["anime", "Japanese animation", "manga-style", "otaku", "animated series"],
    "apple": ["apple", "apple-computing", "iOS", "Mac", "MacOS"],
    "arabic": ["arabic", "Arabic language", "Arabian", "Middle Eastern", "semitic"],
    "arduino": ["arduino", "microcontroller", "embedded systems", "DIY electronics", "maker"],
    "astronomy": ["astronomy", "astrophysics", "celestial", "cosmic", "stellar"],
    "aviation": ["aviation", "aeronautics", "flight", "aircraft", "aerospace"],
    "balinese": ["balinese", "Bali", "Indonesian island", "Hindu-Buddhist", "Balinese culture"],
    "bash": ["bash", "shell scripting", "command line", "terminal", "Unix shell"],
    "bicycles": ["bicycles", "cycling", "bikes", "two-wheelers", "pedal-powered"],
    "bioinformatics": ["bioinformatics", "computational biology", "genomics", "biological data", "biocomputing"],
    "biology": ["biology", "biological", "life sciences", "living organisms", "biological systems"],
    "bitcoin": ["bitcoin", "cryptocurrency", "blockchain", "digital currency", "crypto"],
    "blender": ["blender", "3D modeling", "rendering software", "animation tool", "CGI"],
    "boardgames": ["boardgames", "tabletop games", "board gaming", "strategy games", "family games"],
    "buddhism": ["buddhism", "Buddhist", "dharma", "enlightenment", "meditation"],
    "chemistry": ["chemistry", "chemical", "molecular", "compounds", "reactions"],
    "chess": ["chess", "checkmate", "strategic game", "64 squares", "chess pieces"],
    "christianity": ["christianity", "Christian", "biblical", "gospel", "church"],
    "civicrm": ["civicrm", "CRM software", "constituent management", "nonprofit CRM", "contact management"],
    "codereview": ["codereview", "code review", "peer review", "code inspection", "pull request"],
    "cogsci": ["cogsci", "cognitive science", "cognition", "mental processes", "neurocognitive"],
    "computergraphics": ["computergraphics", "computer graphics", "CGI", "visual computing", "rendering"],
    "cooking": ["cooking", "culinary", "recipes", "cuisine", "food preparation"],
    "cpp": ["cpp", "C++", "C plus plus", "object-oriented", "systems programming"],
    "crypto": ["crypto", "cryptography", "encryption", "security", "ciphers"],
    "cs": ["cs", "computer science", "computing", "algorithms", "programming"],
    "csharp": ["csharp", "C#", "C sharp", ".NET", "Microsoft language"],
    "cybersecurity": ["cybersecurity", "infosec", "security", "cyber defense", "network security"],
    "datascience": ["datascience", "data science", "analytics", "big data", "data analysis"],
    "deutsch": ["deutsch", "German language", "Germanic", "Deutsch", "Deutschland"],
    "earthscience": ["earthscience", "earth science", "geology", "geoscience", "planetary science"],
    "economics": ["economics", "economic", "finance", "markets", "economy"],
    "electronics": ["electronics", "circuits", "electrical", "components", "hardware"],
    "emacs": ["emacs", "text editor", "Elisp", "GNU Emacs", "editor"],
    "engineering": ["engineering", "technical design", "problem-solving", "applied science", "systems design"],
    "fitness": ["fitness", "exercise", "workout", "physical training", "health"],
    "gamedev": ["gamedev", "game development", "game design", "game programming", "video game creation"],
    "gaming": ["gaming", "video games", "gameplay", "gaming culture", "esports"],
    "gardening": ["gardening", "horticulture", "plants", "cultivation", "growing"],
    "genealogy": ["genealogy", "family history", "ancestry", "lineage", "family tree"],
    "graphicdesign": ["graphicdesign", "graphic design", "visual design", "typography", "layout"],
    "hermeneutics": ["hermeneutics", "interpretation", "textual analysis", "exegesis", "biblical interpretation"],
    "hindi": ["hindi", "Hindi language", "Devanagari", "Hindustani", "Indian language"],
    "hinduism": ["hinduism", "Hindu", "Vedic", "dharma", "Sanskrit"],
    "history": ["history", "historical", "past events", "chronology", "heritage"],
    "indonesian": ["indonesian", "Bahasa Indonesia", "Indonesian language", "Malay", "Southeast Asian"],
    "islam": ["islam", "Islamic", "Muslim", "Quran", "Allah"],
    "italian": ["italian", "Italian language", "italiano", "Romance language", "Italy"],
    "java": ["java", "Java programming", "JVM", "object-oriented", "enterprise"],
    "judaism": ["judaism", "Jewish", "Torah", "Hebrew", "synagogue"],
    "khmer": ["khmer", "Cambodian", "Khmer language", "Cambodia", "Southeast Asian"],
    "law": ["law", "legal", "jurisprudence", "legislation", "justice"],
    "linguistics": ["linguistics", "language study", "phonetics", "syntax", "semantics"],
    "literature": ["literature", "literary", "prose", "poetry", "fiction"],
    "math": ["math", "mathematics", "mathematical", "calculus", "algebra"],
    "mathematica": ["mathematica", "Wolfram", "symbolic computation", "computational mathematics", "CAS"],
    "mechanics": ["mechanics", "mechanical", "physics", "dynamics", "kinematics"],
    "medicine": ["medicine", "medical", "healthcare", "clinical", "diagnosis"],
    "money": ["money", "finance", "currency", "economics", "wealth"],
    "movies": ["movies", "cinema", "films", "motion pictures", "cinematography"],
    "music": ["music", "musical", "melody", "rhythm", "composition"],
    "mythology": ["mythology", "myths", "legends", "folklore", "mythological"],
    "networkengineering": ["networkengineering", "network engineering", "networking", "TCP/IP", "routing"],
    "norwegian": ["norwegian", "norsk", "Norwegian language", "Scandinavian", "Nordic"],
    "outdoors": ["outdoors", "outdoor activities", "nature", "hiking", "camping"],
    "pets": ["pets", "animals", "companion animals", "pet care", "domesticated"],
    "philosophy": ["philosophy", "philosophical", "metaphysics", "ethics", "epistemology"],
    "photo": ["photo", "photography", "photographs", "camera", "imaging"],
    "php": ["php", "PHP programming", "web development", "server-side", "scripting"],
    "physics": ["physics", "physical", "quantum", "mechanics", "thermodynamics"],
    "poker": ["poker", "card game", "Texas Hold'em", "gambling", "bluffing"],
    "politics": ["politics", "political", "government", "policy", "democracy"],
    "portuguese": ["portuguese", "português", "Portuguese language", "Lusophone", "Brazil"],
    "puzzling": ["puzzling", "puzzles", "riddles", "brain teasers", "problem-solving"],
    "quant": ["quant", "quantitative", "financial modeling", "algorithmic trading", "mathematical finance"],
    "quantumcomputing": ["quantumcomputing", "quantum computing", "qubits", "superposition", "quantum algorithms"],
    "raspberrypi": ["raspberrypi", "Raspberry Pi", "single-board computer", "IoT", "embedded Linux"],
    "reverseengineering": ["reverseengineering", "reverse engineering", "decompilation", "disassembly", "analysis"],
    "robotics": ["robotics", "robots", "automation", "mechatronics", "AI robotics"],
    "rpg": ["rpg", "role-playing games", "RPGs", "tabletop RPG", "D&D"],
    "russian": ["russian", "русский", "Russian language", "Cyrillic", "Slavic"],
    "scifi": ["scifi", "science fiction", "sci-fi", "futuristic", "speculative"],
    "security": ["security", "cybersecurity", "protection", "safety", "defense"],
    "skeptics": ["skeptics", "skepticism", "critical thinking", "debunking", "rationalism"],
    "softwareengineering": ["softwareengineering", "software engineering", "software development", "SDLC", "programming practices"],
    "sound": ["sound", "audio", "acoustics", "music production", "sound engineering"],
    "space": ["space", "outer space", "cosmos", "universe", "astronomy"],
    "sports": ["sports", "athletics", "competition", "sporting events", "physical activity"],
    "stats": ["stats", "statistics", "statistical analysis", "probability", "data analysis"],
    "swedish": ["swedish", "svenska", "Swedish language", "Nordic", "Scandinavian"],
    "tex": ["tex", "TeX", "LaTeX", "typesetting", "document preparation"],
    "travel": ["travel", "tourism", "journeys", "destinations", "exploration"],
    "turkish": ["turkish", "türkçe", "Turkish language", "Turkic", "Anatolia"],
    "typescript": ["typescript", "TypeScript", "typed JavaScript", "TS", "Microsoft TypeScript"],
    "unix": ["unix", "Unix system", "Linux", "POSIX", "command line"],
    "ux": ["ux", "user experience", "UX design", "usability", "interaction design"],
    "vi": ["vi", "vim", "text editor", "modal editing", "command mode"],
    "webapps": ["webapps", "web applications", "web apps", "SaaS", "browser-based"],
    "wordpress": ["wordpress", "WordPress", "CMS", "blogging platform", "content management"],
    "worldbuilding": ["worldbuilding", "world building", "fictional worlds", "universe creation", "fantasy worlds"],
    "writers": ["writers", "writing", "authors", "creative writing", "storytelling"]
}

# Concept antonyms dictionary - each concept maps to a list of antonyms/opposites
CONCEPT_ANTONYMS = {
    "HTML": ["prose", "cursive", "fiction", "unformatted", "unstructured"],
    "abstract": ["concrete", "non-abstract", "explicit", "practical", "literal"],
    "academic": ["casual", "informal", "stupid", "non-academic", "street-smart"],
    "all-caps": ["lowercase", "no-caps", "sentence-case", "whispered", "subtle"],
    "angry": ["calm", "peaceful", "happy", "content", "serene"],
    "ascii-art": ["fiction", "prose", "normal text", "written words", "standard text"],
    "biology-focused": ["physics-focused", "art-focused", "technology-focused", "non-biological", "inorganic"],
    "boring": ["exciting", "interesting", "engaging", "thrilling", "captivating"],
    "capitalised": ["lowercase", "uncapitalized", "all-lowercase", "minuscule", "small letters"],
    "casual": ["formal", "professional", "official", "academic", "ceremonial"],
    "chemical": ["artistic", "machine-related", "silly", "non-chemical", "art-focused"],
    "chemistry-based": ["artistic", "machine-related", "silly", "non-chemical", "art-focused"],
    "chemistry-focused": ["artistic", "machine-related", "silly", "non-chemical", "art-focused"],
    "chinese": ["english", "western", "non-chinese", "european", "american"],
    "comforting": ["disturbing", "unsettling", "harsh", "cold", "uncomfortable", "scary", "angry"],
    "commanding": ["submissive", "passive", "requesting", "suggesting", "asking"],
    "confused": ["logical", "certain", "sensible", "unconfused", "oriented"],
    "customer-service-roleplay": ["technical documentation", "casual conversation", "academic discourse", "creative writing", "formal report"],
    "czech": ["english", "non-czech", "western", "asian", "american"],
    "dutch": ["english", "non-dutch", "eastern", "asian", "american"],
    "excited": ["calm", "bored", "indifferent", "unenthusiastic", "apathetic"],
    "finnish": ["english", "non-finnish", "southern", "spanish", "mediterranean"],
    "formal": ["casual", "informal", "relaxed", "colloquial", "laid-back"],
    "french": ["english", "german", "non-french", "anglo", "germanic"],
    "french-language": ["english-language", "german-language", "non-french", "anglophone", "germanic"],
    "french-language-only": ["english-only", "multilingual", "non-french", "polyglot", "diverse-language"],
    "fun": ["serious", "boring", "dull", "formal", "tedious"],
    "german": ["english", "french", "non-german", "chinese", "latin"],
    "german-language": ["english", "french", "non-german", "chinese", "latin"],
    "german-language-only": ["english", "french", "non-german", "chinese", "latin", "multilingual"],
    "gibberish": ["coherent", "meaningful", "logical", "clear", "sensible"],
    "greek": ["latin", "modern", "non-greek", "contemporary", "current"],
    "harmful": ["helpful", "beneficial", "safe", "constructive", "positive"],
    "helpful": ["harmful", "useless", "unhelpful", "destructive", "negative"],
    "humorous": ["serious", "solemn", "grave", "humorless", "stern"],
    "imperative": ["optional", "suggestive", "questioning", "passive", "declarative"],
    "japanese": ["western", "english", "non-japanese", "european", "american"],
    "javascript": ["python", "static HTML", "backend", "compiled", "server-side"],
    "jokey": ["serious", "formal", "solemn", "grave", "earnest", "unjokey", "non-jokey"],
    "json-formatted": ["plain text", "unstructured", "prose", "narrative", "free-form"],
    "korean": ["western", "english", "non-korean", "european", "american"],
    "lazy": ["energetic", "motivated", "active", "industrious", "diligent"],
    "literature-focused": ["science-focused", "math-focused", "non-literary"],
    "markdown-formatted": ["plain text", "unformatted", "raw text", "HTML", "rich text"],
    "mathematical": ["literary", "artistic", "non-mathematical"],
    "personality-analysis": ["technical analysis", "objective reporting", "factual description", "impersonal assessment", "statistical analysis"],
    "philosophical": ["practical", "concrete", "mundane", "literal", "factual"],
    "poetic": ["prosaic", "plain", "literal", "straightforward", "unpoetic"],
    "polish": ["english", "non-polish", "western", "germanic", "romance"],
    "python": ["javascript", "compiled", "low-level", "assembly", "machine code"],
    "question-asking": ["statement-making", "declarative", "assertive", "informative", "explaining"],
    "question-laden": ["statement-heavy", "declarative", "assertive", "answer-filled", "informative"],
    "questioning": ["certain", "assertive", "confident", "declarative", "sure"],
    "reassuring": ["worrying", "alarming", "disturbing", "unsettling", "concerning"],
    "reversed": ["forward", "normal", "standard", "regular", "conventional"],
    "rhyming": ["prose", "non-rhyming", "free verse", "unstructured", "arrhythmic"],
    "romanian": ["english", "non-romanian", "western", "germanic", "slavic"],
    "short-sentence-only": ["long sentences", "verbose", "elaborate", "detailed", "complex sentences"],
    "sleepy": ["alert", "awake", "energetic", "wide-awake", "refreshed"],
    "slovak": ["english", "non-slovak", "western", "germanic", "romance"],
    "spanish": ["english", "german", "non-spanish", "anglo", "germanic"],
    "supportive": ["critical", "unsupportive", "discouraging", "negative", "dismissive"],
    "therapeutic": ["harmful", "toxic", "damaging", "destructive", "traumatic"],
    "title-case": ["lowercase", "all-caps", "sentence case", "camelCase", "snake_case"],
    "3dprinting": ["2D printing", "traditional manufacturing", "subtractive manufacturing", "casting", "molding"],
    "academia": ["industry", "practical", "commercial", "non-academic", "vocational"],
    "ai": ["human intelligence", "manual", "non-AI", "traditional computing", "rule-based"],
    "android": ["iOS", "Windows", "desktop", "non-mobile", "Apple"],
    "anime": ["live-action", "western animation", "documentary", "non-animated", "realistic"],
    "apple": ["windows", "android", "linux", "PC", "non-Apple"],
    "arabic": ["english", "western", "latin", "non-arabic", "european"],
    "arduino": ["desktop computing", "cloud computing", "mainframe", "software-only", "high-level"],
    "astronomy": ["geology", "terrestrial", "earthbound", "microscopic", "subatomic"],
    "aviation": ["maritime", "ground transport", "railway", "pedestrian", "nautical"],
    "balinese": ["western", "continental", "mainland", "urban", "industrial"],
    "bash": ["GUI", "graphical", "visual interface", "point-and-click", "Windows CMD"],
    "bicycles": ["cars", "motorized", "automobiles", "powered vehicles", "motorcycles"],
    "bioinformatics": ["pure biology", "wet lab", "field biology", "non-computational", "manual analysis"],
    "biology": ["physics", "chemistry", "geology", "non-living", "inorganic"],
    "bitcoin": ["fiat currency", "traditional banking", "cash", "gold", "physical money"],
    "blender": ["2D graphics", "photo editing", "video editing", "CAD", "hand-drawn"],
    "boardgames": ["video games", "sports", "digital games", "outdoor activities", "electronic games"],
    "buddhism": ["materialism", "atheism", "secularism", "non-religious", "worldly"],
    "chemistry": ["physics", "biology", "mathematics", "non-chemical", "mechanical"],
    "chess": ["checkers", "random games", "chance games", "physical sports", "dice games"],
    "christianity": ["atheism", "secularism", "non-religious", "paganism", "agnosticism"],
    "civicrm": ["spreadsheets", "paper records", "manual tracking", "generic database", "custom solution"],
    "codereview": ["solo coding", "unreviewed code", "direct commit", "self-review", "automated only"],
    "cogsci": ["behaviorism", "pure neuroscience", "philosophy", "folk psychology", "intuition"],
    "computergraphics": ["text-based", "command line", "audio", "non-visual", "console"],
    "cooking": ["raw food", "takeout", "fasting", "processed food", "dining out"],
    "cpp": ["python", "interpreted", "high-level", "scripting", "managed code"],
    "crypto": ["plaintext", "unencrypted", "open", "public", "transparent"],
    "cs": ["humanities", "arts", "literature", "non-technical", "manual labor"],
    "csharp": ["java", "python", "unmanaged", "interpreted", "dynamic"],
    "cybersecurity": ["open access", "unsecured", "vulnerable", "public", "unprotected"],
    "datascience": ["intuition", "guesswork", "qualitative", "anecdotal", "unstructured"],
    "deutsch": ["english", "romance languages", "non-german", "slavic", "asian"],
    "earthscience": ["space science", "astronomy", "astrophysics", "extraterrestrial", "cosmic"],
    "economics": ["non-monetary", "barter", "gift economy", "subsistence", "non-economic"],
    "electronics": ["mechanical", "acoustic", "optical", "pneumatic", "hydraulic"],
    "emacs": ["vim", "nano", "notepad", "IDE", "word processor"],
    "engineering": ["pure science", "theoretical", "artistic", "humanities", "abstract"],
    "fitness": ["sedentary", "inactive", "lazy", "unhealthy", "couch potato"],
    "gamedev": ["game playing", "non-gaming", "serious applications", "productivity software", "business software"],
    "gaming": ["working", "studying", "productive activities", "real life", "outdoor activities"],
    "gardening": ["indoor", "urban", "concrete", "artificial", "synthetic"],
    "genealogy": ["future planning", "present focus", "strangers", "unrelated", "anonymous"],
    "graphicdesign": ["text-only", "audio design", "backend development", "data analysis", "plain text"],
    "hermeneutics": ["literal reading", "surface meaning", "obvious interpretation", "direct meaning", "face value"],
    "hindi": ["english", "western languages", "non-indian", "european", "latin script"],
    "hinduism": ["atheism", "materialism", "secularism", "non-religious", "abrahamic"],
    "history": ["future", "fiction", "contemporary", "present", "speculation"],
    "indonesian": ["english", "european", "non-asian", "western", "continental"],
    "islam": ["atheism", "secularism", "non-religious", "polytheism", "paganism"],
    "italian": ["english", "germanic", "non-romance", "asian", "slavic"],
    "java": ["python", "functional", "procedural", "scripting", "dynamic"],
    "judaism": ["atheism", "paganism", "polytheism", "secularism", "non-religious"],
    "khmer": ["western", "european", "non-asian", "latin", "germanic"],
    "law": ["anarchy", "lawlessness", "chaos", "disorder", "vigilante"],
    "linguistics": ["non-verbal", "silence", "gibberish", "meaningless", "inarticulate"],
    "literature": ["technical writing", "scientific papers", "mathematics", "code", "data"],
    "math": ["literature", "art", "qualitative", "subjective", "non-mathematical"],
    "mathematica": ["manual calculation", "pen and paper", "mental math", "approximate", "qualitative"],
    "mechanics": ["electronics", "software", "abstract", "theoretical", "quantum"],
    "medicine": ["disease", "illness", "harm", "poison", "injury"],
    "money": ["barter", "gift", "free", "volunteer", "non-monetary"],
    "movies": ["books", "radio", "podcasts", "still images", "live theater"],
    "music": ["silence", "noise", "spoken word", "visual art", "literature"],
    "mythology": ["history", "fact", "science", "reality", "documentation"],
    "networkengineering": ["standalone", "offline", "disconnected", "isolated", "local-only"],
    "norwegian": ["english", "southern", "mediterranean", "tropical", "equatorial"],
    "outdoors": ["indoors", "urban", "virtual", "online", "domestic"],
    "pets": ["wild animals", "livestock", "pests", "zoo animals", "feral"],
    "philosophy": ["practical", "concrete", "empirical", "factual", "mundane"],
    "photo": ["audio", "text", "video", "animation", "illustration"],
    "php": ["compiled languages", "desktop applications", "system programming", "embedded", "mobile apps"],
    "physics": ["metaphysics", "philosophy", "spirituality", "mysticism", "pseudoscience"],
    "poker": ["solitaire", "cooperative games", "deterministic games", "perfect information", "chess"],
    "politics": ["apolitical", "anarchy", "personal", "private", "individual"],
    "portuguese": ["english", "germanic", "asian", "slavic", "non-romance"],
    "puzzling": ["straightforward", "obvious", "simple", "clear", "direct"],
    "quant": ["qualitative", "intuitive", "subjective", "artistic", "emotional"],
    "quantumcomputing": ["classical computing", "analog", "mechanical", "manual", "abacus"],
    "raspberrypi": ["mainframe", "supercomputer", "cloud", "desktop PC", "server farm"],
    "reverseengineering": ["forward engineering", "design", "creation", "building", "construction"],
    "robotics": ["manual labor", "human-operated", "organic", "biological", "natural"],
    "rpg": ["sports games", "puzzle games", "racing games", "strategy games", "shooters"],
    "russian": ["english", "western", "latin", "romance", "germanic"],
    "scifi": ["historical fiction", "contemporary", "realistic", "documentary", "non-fiction"],
    "security": ["vulnerability", "openness", "exposure", "risk", "insecurity"],
    "skeptics": ["believers", "faithful", "credulous", "trusting", "gullible"],
    "softwareengineering": ["hardware engineering", "manual processes", "ad-hoc coding", "unstructured", "cowboy coding"],
    "sound": ["silence", "visual", "tactile", "text", "mute"],
    "space": ["earth", "terrestrial", "underground", "oceanic", "atmospheric"],
    "sports": ["sedentary", "mental activities", "board games", "reading", "meditation"],
    "stats": ["intuition", "guesswork", "anecdotal", "qualitative", "subjective"],
    "swedish": ["english", "southern", "mediterranean", "tropical", "latin"],
    "tex": ["word processor", "WYSIWYG", "plain text", "handwriting", "typewriter"],
    "travel": ["staying home", "local", "stationary", "sedentary", "homebound"],
    "turkish": ["english", "european", "indo-european", "slavic", "germanic"],
    "typescript": ["untyped javascript", "dynamically typed", "weakly typed", "assembly", "machine code"],
    "unix": ["windows", "DOS", "mainframe", "proprietary", "closed-source"],
    "ux": ["backend", "database", "system design", "infrastructure", "developer experience"],
    "vi": ["emacs", "nano", "GUI editors", "word processors", "IDEs"],
    "webapps": ["desktop apps", "mobile apps", "command line", "embedded", "firmware"],
    "wordpress": ["static HTML", "custom CMS", "no CMS", "hand-coded", "from scratch"],
    "worldbuilding": ["real world", "documentary", "non-fiction", "reporting", "journalism"],
    "writers": ["readers", "critics", "editors", "publishers", "non-writers"]
}

def get_random_synonym(concept: str) -> str:
    """Get a random synonym for the given concept, falling back to the original if not found."""
    synonyms = CONCEPT_SYNONYMS.get(concept, [concept])
    return random.choice(synonyms)

def get_random_antonym(concept: str) -> str:
    """Get a random antonym for the given concept. Returns None if no antonyms are defined."""
    antonyms = CONCEPT_ANTONYMS.get(concept, None)
    if antonyms:
        return random.choice(antonyms)
    return None

def plot_validation_curves(validation_logs: List[Dict], save_path: str):
    """Generate and save a dedicated validation loss plot."""
    if not validation_logs:
        return

    val_steps = [log['step'] for log in validation_logs]
    val_beh_losses = [log['avg_beh_loss'] for log in validation_logs]
    val_obf_losses = [log['avg_obf_loss'] for log in validation_logs]

    # Create 2x1 subplot layout for validation losses
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Validation behavior loss
    ax1.plot(val_steps, val_beh_losses, 'o-', label='Validation Behavior Loss',
             color='lightblue', markersize=4, linewidth=2)
    ax1.set_title('Validation Behavior Loss Over Training')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Behavior Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Validation obfuscation loss
    ax2.plot(val_steps, val_obf_losses, 'o-', label='Validation Obfuscation Loss',
             color='orange', markersize=4, linewidth=2)
    ax2.set_title('Validation Obfuscation Loss Over Training')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Obfuscation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save PNG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # Save PDF
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    # Create combined validation plot
    fig, ax = plt.subplots(figsize=(12, 6))

    if val_beh_losses:
        ax.plot(val_steps, val_beh_losses, 'o-', label='Validation Behavior Loss',
               color='lightblue', markersize=4, linewidth=2)

    if val_obf_losses:
        ax.plot(val_steps, val_obf_losses, 'o-', label='Validation Obfuscation Loss',
               color='orange', markersize=4, linewidth=2)

    ax.set_title('Validation Loss Curves')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    combined_val_path = save_path.replace('.png', '_combined.png')
    # Save PNG
    plt.savefig(combined_val_path, dpi=300, bbox_inches='tight')
    # Save PDF
    combined_pdf_path = combined_val_path.replace('.png', '.pdf')
    plt.savefig(combined_pdf_path, bbox_inches='tight')
    plt.close()

    print(f"Validation plots saved to {save_path} and {combined_val_path}")

    # Upload plots to wandb if available
    try:
        import wandb
        if wandb.run is not None:
            # Upload PNG files
            wandb.save(str(save_path), base_path=str(Path(save_path).parent))
            wandb.save(str(combined_val_path), base_path=str(Path(combined_val_path).parent))

            # Upload PDF files
            pdf_path = save_path.replace('.png', '.pdf')
            combined_pdf_path = combined_val_path.replace('.png', '.pdf')
            if Path(pdf_path).exists():
                wandb.save(str(pdf_path), base_path=str(Path(pdf_path).parent))
            if Path(combined_pdf_path).exists():
                wandb.save(str(combined_pdf_path), base_path=str(Path(combined_pdf_path).parent))

            print("Uploaded plots to wandb")
    except Exception as e:
        print(f"Could not upload plots to wandb: {e}")

def plot_validation_by_data_type(validation_logs: List[Dict], save_path: str):
    """Generate validation loss plots separated by abstract data type patterns."""
    if not validation_logs or not any('by_data_type' in log for log in validation_logs):
        print("No data type information available in validation logs")
        return

    # Collect all data types and organize data
    all_data_types = set()
    for log in validation_logs:
        if 'by_data_type' in log:
            all_data_types.update(log['by_data_type'].keys())

    all_data_types = sorted(list(all_data_types))

    # Prepare data for plotting
    steps = []
    obf_losses_by_type = {dt: [] for dt in all_data_types}
    beh_losses_by_type = {dt: [] for dt in all_data_types}

    for log in validation_logs:
        if 'by_data_type' not in log:
            continue
        steps.append(log['step'])
        for dt in all_data_types:
            if dt in log['by_data_type']:
                obf_losses_by_type[dt].append(log['by_data_type'][dt]['obf_loss'])
                beh_losses_by_type[dt].append(log['by_data_type'][dt]['beh_loss'])
            else:
                obf_losses_by_type[dt].append(None)
                beh_losses_by_type[dt].append(None)

    # Create plot with subplots for obf and beh losses
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Color palette
    colors = plt.cm.tab20(range(len(all_data_types)))

    # Plot obfuscation losses
    for i, dt in enumerate(all_data_types):
        # Filter out None values
        valid_indices = [j for j, v in enumerate(obf_losses_by_type[dt]) if v is not None]
        if valid_indices:
            valid_steps = [steps[j] for j in valid_indices]
            valid_losses = [obf_losses_by_type[dt][j] for j in valid_indices]
            ax1.plot(valid_steps, valid_losses, 'o-', label=dt,
                    color=colors[i], markersize=3, linewidth=1.5, alpha=0.8)

    ax1.set_title('Validation Obfuscation Loss by Data Type Pattern')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Obfuscation Loss')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot behavior losses
    for i, dt in enumerate(all_data_types):
        # Filter out None values
        valid_indices = [j for j, v in enumerate(beh_losses_by_type[dt]) if v is not None]
        if valid_indices:
            valid_steps = [steps[j] for j in valid_indices]
            valid_losses = [beh_losses_by_type[dt][j] for j in valid_indices]
            ax2.plot(valid_steps, valid_losses, 'o-', label=dt,
                    color=colors[i], markersize=3, linewidth=1.5, alpha=0.8)

    ax2.set_title('Validation Behavior Loss by Data Type Pattern')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Behavior Loss')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"Data type validation plots saved to {save_path}")

def plot_validation_per_probe(validation_logs: List[Dict], output_dir: str):
    """Generate per-probe validation loss plots with abstract clue notation."""
    if not validation_logs or not any('by_probe' in log for log in validation_logs):
        print("No probe-specific information available in validation logs")
        return

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Collect all probes
    all_probes = set()
    for log in validation_logs:
        if 'by_probe' in log:
            all_probes.update(log['by_probe'].keys())

    all_probes = sorted(list(all_probes))

    # Generate plot for each probe
    for probe_name in all_probes:
        # Collect all data type patterns for this probe and normalize them
        all_patterns_raw = set()
        for log in validation_logs:
            if 'by_probe' in log and probe_name in log['by_probe']:
                all_patterns_raw.update(log['by_probe'][probe_name].keys())

        # Normalize patterns: replace specific clue concepts with abstract notation
        pattern_mapping = {}  # normalized -> list of raw patterns
        for raw_pattern in all_patterns_raw:
            parts = raw_pattern.split('_')
            if len(parts) >= 7 and parts[0] == 'clue' and parts[2] == 'data' and parts[4] == 'probe':
                clue_concept = parts[1]
                # Determine if clue is X (same as probe), N (no clue), or Y (different)
                if clue_concept == probe_name:
                    abstract_clue = 'X'
                elif clue_concept == 'N' or clue_concept == 'none':
                    abstract_clue = 'N'
                else:
                    abstract_clue = 'Y'

                # Reconstruct pattern with abstract clue
                normalized = f'clue_{abstract_clue}_' + '_'.join(parts[2:])

                if normalized not in pattern_mapping:
                    pattern_mapping[normalized] = []
                pattern_mapping[normalized].append(raw_pattern)
            else:
                # Keep pattern as is if it doesn't match expected format
                if raw_pattern not in pattern_mapping:
                    pattern_mapping[raw_pattern] = []
                pattern_mapping[raw_pattern].append(raw_pattern)

        all_patterns = sorted(list(pattern_mapping.keys()))

        if not all_patterns:
            continue

        # Prepare data - aggregate across all raw patterns that map to same normalized pattern
        steps = []
        obf_losses_by_pattern = {pattern: [] for pattern in all_patterns}
        beh_losses_by_pattern = {pattern: [] for pattern in all_patterns}

        for log in validation_logs:
            if 'by_probe' not in log or probe_name not in log['by_probe']:
                continue
            steps.append(log['step'])

            for normalized_pattern in all_patterns:
                raw_patterns = pattern_mapping[normalized_pattern]

                # Collect losses from all raw patterns that map to this normalized pattern
                obf_losses = []
                beh_losses = []
                for raw_pattern in raw_patterns:
                    if raw_pattern in log['by_probe'][probe_name]:
                        obf_losses.append(log['by_probe'][probe_name][raw_pattern]['obf_loss'])
                        beh_losses.append(log['by_probe'][probe_name][raw_pattern]['beh_loss'])

                # Average if we have multiple values, otherwise None
                if obf_losses:
                    obf_losses_by_pattern[normalized_pattern].append(np.mean(obf_losses))
                    beh_losses_by_pattern[normalized_pattern].append(np.mean(beh_losses))
                else:
                    obf_losses_by_pattern[normalized_pattern].append(None)
                    beh_losses_by_pattern[normalized_pattern].append(None)

        if not steps:
            continue

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Color palette
        colors = plt.cm.tab20(range(len(all_patterns)))

        # Plot obfuscation losses
        for i, pattern in enumerate(all_patterns):
            valid_indices = [j for j, v in enumerate(obf_losses_by_pattern[pattern]) if v is not None]
            if valid_indices:
                valid_steps = [steps[j] for j in valid_indices]
                valid_losses = [obf_losses_by_pattern[pattern][j] for j in valid_indices]
                # Truncate long labels for display
                display_label = pattern if len(pattern) < 50 else pattern[:47] + '...'
                ax1.plot(valid_steps, valid_losses, 'o-', label=display_label,
                        color=colors[i % len(colors)], markersize=3, linewidth=1.5, alpha=0.8)

        ax1.set_title(f'Validation Obfuscation Loss for Probe: {probe_name}')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Obfuscation Loss')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax1.grid(True, alpha=0.3)

        # Plot behavior losses
        for i, pattern in enumerate(all_patterns):
            valid_indices = [j for j, v in enumerate(beh_losses_by_pattern[pattern]) if v is not None]
            if valid_indices:
                valid_steps = [steps[j] for j in valid_indices]
                valid_losses = [beh_losses_by_pattern[pattern][j] for j in valid_indices]
                # Truncate long labels for display
                display_label = pattern if len(pattern) < 50 else pattern[:47] + '...'
                ax2.plot(valid_steps, valid_losses, 'o-', label=display_label,
                        color=colors[i % len(colors)], markersize=3, linewidth=1.5, alpha=0.8)

        ax2.set_title(f'Validation Behavior Loss for Probe: {probe_name}')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Behavior Loss')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{probe_name}_validation.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()

        print(f"Validation plot for probe '{probe_name}' saved to {save_path}")

    print(f"Generated {len(all_probes)} per-probe validation plots in {output_dir}")

def plot_validation_by_concept_and_type(validation_logs: List[Dict], output_dir: str):
    """Generate validation loss plots for clue_X_data_X_probe_X_[type] patterns, one plot per type with lines per concept."""
    if not validation_logs or not any('by_probe' in log for log in validation_logs):
        print("No probe-specific information available in validation logs")
        return

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Collect all patterns and organize by type
    patterns_by_type = {
        'topical': {},  # concept -> data over time
        'vanilla': {},
        'ultrachat': {}
    }

    # First pass: collect all clue_X_data_X_probe_X_type patterns
    # Note: data_X means data for the same concept, represented as X (not literally X)
    for log in validation_logs:
        if 'by_probe' not in log:
            continue

        for probe_name, probe_data in log['by_probe'].items():
            for pattern in probe_data.keys():
                # Check if pattern matches clue_concept_data_X_probe_concept_type format
                # where concept in clue position matches concept in probe position
                parts = pattern.split('_')

                # Look for patterns like clue_concept_data_X_probe_concept_type
                if len(parts) >= 7 and parts[0] == 'clue' and parts[2] == 'data' and parts[4] == 'probe':
                    clue_concept = parts[1]
                    data_marker = parts[3]  # This will be 'X' or 'Y'
                    probe_concept = parts[5]
                    pattern_type = parts[6] if len(parts) > 6 else None

                    # Check if clue and probe match, and data is X (meaning same concept)
                    if clue_concept == probe_concept == probe_name and data_marker == 'X':
                        if pattern_type in patterns_by_type:
                            if probe_name not in patterns_by_type[pattern_type]:
                                patterns_by_type[pattern_type][probe_name] = []

    # Debug: print what we found
    print(f"Found patterns by type:")
    for ptype, concepts in patterns_by_type.items():
        if concepts:
            print(f"  {ptype}: {len(concepts)} concepts - {sorted(concepts.keys())[:5]}...")

    # Second pass: collect data for each pattern
    for pattern_type, concepts in patterns_by_type.items():
        if not concepts:
            continue

        print(f"Processing {pattern_type} patterns with {len(concepts)} concepts")

        # Prepare data structure
        steps = []
        obf_losses_by_concept = {concept: [] for concept in concepts}
        beh_losses_by_concept = {concept: [] for concept in concepts}

        for log in validation_logs:
            if 'by_probe' not in log:
                continue

            steps.append(log['step'])

            for concept in concepts:
                # Look for the specific pattern for this concept
                # Pattern uses 'X' as placeholder for same concept
                pattern = f'clue_{concept}_data_X_probe_{concept}_{pattern_type}'

                if concept in log['by_probe'] and pattern in log['by_probe'][concept]:
                    obf_losses_by_concept[concept].append(log['by_probe'][concept][pattern]['obf_loss'])
                    beh_losses_by_concept[concept].append(log['by_probe'][concept][pattern]['beh_loss'])
                else:
                    obf_losses_by_concept[concept].append(None)
                    beh_losses_by_concept[concept].append(None)

        if not steps:
            print(f"No steps found for {pattern_type}")
            continue

        # Create plot for this type
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Color palette
        sorted_concepts = sorted(concepts.keys())
        colors = plt.cm.tab20(range(len(sorted_concepts)))

        # Plot obfuscation losses
        for i, concept in enumerate(sorted_concepts):
            valid_indices = [j for j, v in enumerate(obf_losses_by_concept[concept]) if v is not None]
            if valid_indices:
                valid_steps = [steps[j] for j in valid_indices]
                valid_losses = [obf_losses_by_concept[concept][j] for j in valid_indices]
                ax1.plot(valid_steps, valid_losses, 'o-', label=concept,
                        color=colors[i % len(colors)], markersize=3, linewidth=1.5, alpha=0.8)

        ax1.set_title(f'Validation Obfuscation Loss - {pattern_type.capitalize()} (clue_X_data_X_probe_X_{pattern_type})')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Obfuscation Loss')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot behavior losses
        for i, concept in enumerate(sorted_concepts):
            valid_indices = [j for j, v in enumerate(beh_losses_by_concept[concept]) if v is not None]
            if valid_indices:
                valid_steps = [steps[j] for j in valid_indices]
                valid_losses = [beh_losses_by_concept[concept][j] for j in valid_indices]
                ax2.plot(valid_steps, valid_losses, 'o-', label=concept,
                        color=colors[i % len(colors)], markersize=3, linewidth=1.5, alpha=0.8)

        ax2.set_title(f'Validation Behavior Loss - {pattern_type.capitalize()} (clue_X_data_X_probe_X_{pattern_type})')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Behavior Loss')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'validation_clue_X_data_X_probe_X_{pattern_type}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()

        print(f"Validation plot for clue_X_data_X_probe_X_{pattern_type} saved to {save_path}")

    print(f"Generated validation plots by concept and type in {output_dir}")

def load_concept_probes(probe_checkpoints_path: str) -> Dict[str, object]:
    """Load trained concept probes from checkpoint directory."""
    print(f"Loading concept probes from: {probe_checkpoints_path}")
    concept_to_probe = {}
    probe_files = glob.glob(os.path.join(probe_checkpoints_path, "*.pkl"))
    print(f"Found {len(probe_files)} probe files")

    for probe_file in probe_files:
        concept = Path(probe_file).stem
        print(f"Loading probe for concept: {concept}")
        try:
            with open(probe_file, 'rb') as f:
                probe = pickle.load(f)
            concept_to_probe[concept] = probe
            print(f"Successfully loaded probe for {concept}")
        except (IOError, pickle.PickleError) as e:
            print(f"Error loading probe file {probe_file}: {e}")
            continue

    print(f"Finished loading {len(concept_to_probe)} concept probes")
    return concept_to_probe

def evaluate_validation_set(
    model,  # HFHardPromptedWithSelectableLayers
    val_data: List,
    target_layers: List[int],
    behaviour_lambda: float,
    step_num: int,
    logger,
    validation_metadata: List[Dict] = None,
    concept_to_probe: Dict = None
) -> Dict:
    """Evaluate model on validation set and return loss metrics."""
    model.eval()
    total_obf_loss = 0.0
    total_beh_loss = 0.0
    total_samples = 0

    # Track losses by data type and probe
    by_data_type = {}
    by_probe = {}

    with torch.no_grad():
        for idx, data_item in enumerate(val_data):
            probe, prompt_text, response_text, token_ids, label, use_behavior_loss = data_item

            # Forward pass
            if token_ids is not None:
                behavior_output = model.forward_from_string_and_ids(
                    input_text=[prompt_text],
                    target_ids=torch.tensor(token_ids, device=model.device).unsqueeze(0),
                    target_attn_mask=torch.ones_like(torch.tensor(token_ids, device=model.device)).unsqueeze(0),
                    add_chat_template=True,
                    use_tunable_params=False,
                    layers_to_probe=target_layers
                )
            else:
                behavior_output = model.forward_from_string(
                    input_text=[prompt_text],
                    target_text=[response_text],
                    add_chat_template=True,
                    use_tunable_params=False,
                    layers_to_probe=target_layers
                )

            # Get target tokens
            if token_ids is not None:
                behavior_target_input_ids = torch.tensor(token_ids, device=model.device).unsqueeze(0)
            else:
                behavior_target_input_ids, _ = model.tokenize(
                    text=[response_text],
                    add_chat_template=False,
                    add_special_tokens=False,
                    max_length=behavior_output.target_logits.size(1),
                    pad_to_max_length=True
                )

            # Calculate behavior loss
            if use_behavior_loss:
                beh_losses = F.cross_entropy(
                    behavior_output.target_logits.transpose(1, 2),
                    behavior_target_input_ids,
                    reduction="none"
                )
                beh_losses *= behavior_output.loss_mask.float()
                num_valid_tokens = behavior_output.loss_mask.float().sum(dim=-1)
                behavior_loss = beh_losses.sum(dim=-1) / (num_valid_tokens + 1e-8)
            else:
                behavior_loss = torch.tensor(0.0, device=behavior_output.target_logits.device)

            # Calculate obfuscation loss (handle both single and dual probes)
            if isinstance(probe, tuple):
                # Dual probe case
                probe_x, probe_y = probe
                label_x, label_y = label

                scores_x = probe_x.predict(reps=behavior_output.target_reps)
                scores_y = probe_y.predict(reps=behavior_output.target_reps)

                ideal_scores_x = torch.tensor([label_x], device=model.device, dtype=torch.float16)
                ideal_scores_y = torch.tensor([label_y], device=model.device, dtype=torch.float16)

                obf_loss_x = F.mse_loss(scores_x, ideal_scores_x, reduction="none")
                obf_loss_y = F.mse_loss(scores_y, ideal_scores_y, reduction="none")

                # Average the two losses
                obf_loss = (obf_loss_x + obf_loss_y) / 2
            else:
                # Single probe case
                scores = probe.predict(reps=behavior_output.target_reps)
                ideal_scores = torch.tensor([label], device=model.device, dtype=torch.float16)
                obf_loss = F.mse_loss(scores, ideal_scores, reduction="none")

            obf_loss_val = obf_loss.mean().item()
            beh_loss_val = behavior_loss.mean().item() if hasattr(behavior_loss, 'mean') else behavior_loss.item()

            total_obf_loss += obf_loss_val
            total_beh_loss += beh_loss_val
            total_samples += 1

            # Track by data type and probe if metadata is available
            if validation_metadata and idx < len(validation_metadata):
                meta = validation_metadata[idx]
                data_type = meta.get('data_type', 'unknown')
                probe_name = meta.get('probe_name', 'unknown')

                # Track by abstract data type
                if data_type not in by_data_type:
                    by_data_type[data_type] = {
                        'obf_loss': 0.0,
                        'beh_loss': 0.0,
                        'count': 0
                    }
                by_data_type[data_type]['obf_loss'] += obf_loss_val
                by_data_type[data_type]['beh_loss'] += beh_loss_val
                by_data_type[data_type]['count'] += 1

                # Track by probe with full descriptive label
                if probe_name != 'unknown':
                    # Create full descriptive label by replacing X/Y/N with actual concepts
                    full_label = data_type

                    # Parse the clue from prompt if it exists
                    clue_concept = None
                    if "You are being probed for" in prompt_text:
                        import re
                        match = re.search(r"You are being probed for '([^']+)'", prompt_text)
                        if match:
                            clue_text = match.group(1)
                            # Map clue text back to concept using synonyms or direct match
                            if concept_to_probe:
                                for concept in concept_to_probe.keys():
                                    if clue_text == concept or clue_text in CONCEPT_SYNONYMS.get(concept, []):
                                        clue_concept = concept
                                        break

                    # Build the full descriptive label
                    if clue_concept:
                        full_label = data_type.replace('clue_X', f'clue_{clue_concept}')
                    else:
                        full_label = data_type.replace('clue_N', 'clue_none')

                    # For dual probe cases
                    if isinstance(probe, tuple):
                        probe_x_name, probe_y_name = probe_name.strip('()').split(',')
                        full_label = full_label.replace('probe_XY', f'probe_{probe_x_name}_{probe_y_name}')
                        full_label = full_label.replace('data_XY', f'data_{probe_x_name}_{probe_y_name}')

                        # Track for both probes
                        for p_name in [probe_x_name, probe_y_name]:
                            if p_name not in by_probe:
                                by_probe[p_name] = {}
                            if full_label not in by_probe[p_name]:
                                by_probe[p_name][full_label] = {
                                    'obf_loss': 0.0,
                                    'beh_loss': 0.0,
                                    'count': 0
                                }
                            by_probe[p_name][full_label]['obf_loss'] += obf_loss_val
                            by_probe[p_name][full_label]['beh_loss'] += beh_loss_val
                            by_probe[p_name][full_label]['count'] += 1
                    else:
                        # Single probe case
                        full_label = full_label.replace('probe_X', f'probe_{probe_name}')
                        full_label = full_label.replace('probe_Y', f'probe_{probe_name}')

                        # Try to infer data concept from data type pattern
                        if 'data_X' in data_type or 'data_Y' in data_type:
                            # This requires more context, for now keep as is
                            pass

                        if probe_name not in by_probe:
                            by_probe[probe_name] = {}
                        if full_label not in by_probe[probe_name]:
                            by_probe[probe_name][full_label] = {
                                'obf_loss': 0.0,
                                'beh_loss': 0.0,
                                'count': 0
                            }
                        by_probe[probe_name][full_label]['obf_loss'] += obf_loss_val
                        by_probe[probe_name][full_label]['beh_loss'] += beh_loss_val
                        by_probe[probe_name][full_label]['count'] += 1

    model.train()

    # Calculate averages for by_data_type
    for dt in by_data_type:
        if by_data_type[dt]['count'] > 0:
            by_data_type[dt]['obf_loss'] /= by_data_type[dt]['count']
            by_data_type[dt]['beh_loss'] /= by_data_type[dt]['count']

    # Calculate averages for by_probe
    for probe_name in by_probe:
        for label in by_probe[probe_name]:
            if by_probe[probe_name][label]['count'] > 0:
                by_probe[probe_name][label]['obf_loss'] /= by_probe[probe_name][label]['count']
                by_probe[probe_name][label]['beh_loss'] /= by_probe[probe_name][label]['count']

    validation_result = {
        'step': step_num,
        'avg_obf_loss': total_obf_loss / total_samples if total_samples > 0 else 0,
        'avg_beh_loss': total_beh_loss / total_samples if total_samples > 0 else 0,
        'num_samples': total_samples,
        'by_data_type': by_data_type,
        'by_probe': by_probe
    }

    logger.print(f"Validation at step {step_num}: obf_loss={validation_result['avg_obf_loss']:.4f}, beh_loss={validation_result['avg_beh_loss']:.4f}")

    return validation_result
