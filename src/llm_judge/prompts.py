"""
Zero-shot and few-shot prompt templates for LLM judge evaluation.

Few-shot examples are real samples extracted from the SSD-2026 training data.
"""

# ── Task 1: Supportive vs Non-Supportive ────────────────────────────────────

TASK1_ZERO_SHOT = """Classify the following social media comment as either "Supportive" or "Non-Supportive".
A comment is Supportive if it expresses encouragement, care, solidarity, or help toward someone.
A comment promoting violence is NOT Supportive even if it appears positive.

Comment: {text}
Label:"""

TASK1_FEW_SHOT = """Classify each social media comment as either "Supportive" or "Non-Supportive".
A comment is Supportive if it expresses encouragement, care, solidarity, or help toward someone.
A comment promoting violence is NOT Supportive even if it appears positive.

Examples:

Comment: heart breaking crying face free palestine please stop killing children innocent people
Label: Supportive

Comment: cant stand cops hate woke protesters even
Label: Non-Supportive

Comment: love little boy stay strong red heartred heartred heartred heart free palestine
Label: Supportive

Comment: said push say need vacation enjoy something like understanding mean like lazy
Label: Non-Supportive

Comment: scotland support palestine sorry suffering israel
Label: Supportive

Comment: full funding israel keeping hamas rich part netenyahus strategy read words
Label: Non-Supportive

Comment: support palestine indonesia strong palestine believe win
Label: Supportive

Comment: judah traded come ne africa renamed middle east know like everyone else negroes helped hide identity
Label: Non-Supportive

Now classify:
Comment: {text}
Label:"""

# ── Task 2: Individual vs Group ─────────────────────────────────────────────

TASK2_ZERO_SHOT = """This social media comment has been identified as supportive. Classify whether the support is directed at an "Individual" or a "Group".
- Individual: support directed at a specific person
- Group: support directed at a community, group, or cause

Comment: {text}
Label:"""

TASK2_FEW_SHOT = """This social media comment has been identified as supportive. Classify whether the support is directed at an "Individual" or a "Group".
- Individual: support directed at a specific person
- Group: support directed at a community, group, or cause

Examples:

Comment: may god bless protect happiness wish could hug little sweetheart cherry blossomfolded handsmediumlight skin tone
Label: Individual

Comment: free palestine worry allah help family loudly crying faceloudly crying face
Label: Group

Comment: thinking precious baby lot happy relieved recovering shock thank god still family red heartred heartred heart
Label: Individual

Comment: love pekanbru indonesia palestina always support keep strong
Label: Group

Comment: thanks sharing update child happy see okay
Label: Individual

Comment: thing really makes cry pakistan ya allah reham palestine syrian condition
Label: Group

Comment: take nothing man true legend leader messi opinion greatest liverpool fan guy grown right top greatest timered heart
Label: Individual

Comment: instead blame others please pray palestine crying face
Label: Group

Now classify:
Comment: {text}
Label:"""

# ── Task 3: Community classification ────────────────────────────────────────

TASK3_ZERO_SHOT = """This social media comment expresses group support. Classify which community it supports.
Choose exactly one: "Nation", "Other", "LGBTQ", "Black Community", "Religion", "Women"

- Nation: support for a country or national identity
- Other: support for a group not in the other categories (e.g. children, firefighters, general humanity)
- LGBTQ: support for LGBTQ+ community
- Black Community: support for Black/African American community
- Religion: support for a religious group or faith community
- Women: support for women's rights or empowerment

Comment: {text}
Label:"""

TASK3_FEW_SHOT = """This social media comment expresses group support. Classify which community it supports.
Choose exactly one: "Nation", "Other", "LGBTQ", "Black Community", "Religion", "Women"

- Nation: support for a country or national identity
- Other: support for a group not in the other categories (e.g. children, firefighters, general humanity)
- LGBTQ: support for LGBTQ+ community
- Black Community: support for Black/African American community
- Religion: support for a religious group or faith community
- Women: support for women's rights or empowerment

Examples:

Comment: palestine always heart endure n persevere
Label: Nation

Comment: crying face help cry palestinian children wish could go help
Label: Other

Comment: im christian still support lgbtq rights marriage want equality
Label: LGBTQ

Comment: planned parenthood supports black lives matter stop free black protester stick
Label: Black Community

Comment: muslim leaders please help save muslim countries tortured dont close eyes
Label: Religion

Comment: comments nutshell guy pissed treat women like human beings support violence women equal rights
Label: Women

Comment: never grow old killed kidnapped jailed bombed since child pray palestin red heart palestin
Label: Nation

Comment: religion race matter innocent people dying stop war
Label: Other

Comment: something never understand straight couples make public lgbtq couples even hold hands without getting disapproving stares
Label: LGBTQ

Comment: media shows side blm peaceful side
Label: Black Community

Comment: much love muslim palastinian brothers russian eastern orthadox faith
Label: Religion

Comment: forget wish childless women happy mother day comes around
Label: Women

Now classify:
Comment: {text}
Label:"""

# ── Combined: All 3 tasks in one prompt ─────────────────────────────────────

COMBINED_FEW_SHOT = """Classify the following social media comment on three hierarchical tasks. Respond ONLY with a JSON object, no other text.
 
Task 1 — Support Detection: Is the comment "Supportive" or "Non-Supportive"?
 
A comment is Supportive if it expresses encouragement, care, admiration, help, or solidarity TOWARD a specific person, group, or cause. The support must have a clear, identifiable target.
 
A comment that explicitly promotes violence or harm is always Non-Supportive, even if it names a group.
 
Supportive — the comment does one or more of these toward a specific target:
- Expresses solidarity or advocacy ("we stand with Palestine", "support black people")
- Offers encouragement to someone in hardship ("stay strong", "praying for you")
- Directs personal admiration, love, or defence at a person ("I love ronaldo", "people hate him but he is great")
- Makes a plea, prayer, or wish for a specific person or group ("oh allah help syria", "may god protect the children")
- Offers to help or expresses a desire to help ("wish I could go help", "please donate")
- Advocates for a group's rights or wellbeing ("equal rights for women", "support lgbtq rights")
 
Non-Supportive — the comment does NOT direct support at a specific target. This includes:
- General reactions to content ("great video", "so inspiring")
- Religious preaching or doctrine not directed at a person or group ("God is supreme", "Jesus saves")
- Personal emotional reactions, even about tragic events ("tears come to my eyes", "heart bursting pain watching") — expressing YOUR emotional response is not directing support
- Generic or objective praise of a public figure ("true legend", "steve jobs is inspiring") — describing qualities ≠ directing personal support. Contrast with "I love ronaldo" which IS Supportive
- Motivational content without a specific target ("never stop dreaming", "stay hungry")
- Commentary or observation without directing support ("men care about women's success", "war threatens the whole world")
- Merely stating a side or opinion without actively offering support, prayer, or solidarity ("people support palestine", "I'm on their side")
- Political opposition AGAINST someone ("resign trump", "stop putin") — opposing ≠ supporting
- Criticizing or debating a movement ("BLM funded by clinton", "movement is racist")
- Fan debate or sports discussion ("messi is the greatest", "people finally see the true ronaldo")
 
Key distinction: The comment must actively direct encouragement, care, admiration, help, or solidarity toward a target — not merely react, narrate, state an opinion, or express personal emotion about a situation.
 
Task 2 — Target Type (only if Supportive): "Individual" or "Group"
- Individual: support directed at a specific person (named or identifiable from context)
- Group: support directed at a community, nation, or cause
- If Non-Supportive → "No"
 
Task 3 — Targeted Group (only if Group): Which community is supported?
Choose ONE from: "Nation", "Other", "LGBTQ", "Black Community", "Religion", "Women"
 
- Nation: support directed at a country, its people, or national sovereignty ("free Palestine", "support Ukraine", "praying for Syria"). Use when the comment names a specific country and directs support at that country or its people as a national group.
- Other: support directed at a group not in the other categories — children, innocent people, humanity, peace, or general welfare. Use when the support targets people as humans or innocents rather than as members of a specific national, religious, or identity group.
- LGBTQ: support for the LGBTQ+ community or LGBTQ rights.
- Black Community: support for the Black/African American community or racial justice for Black people.
- Religion: support for a religious community framed around religious identity ("help our muslim brothers and sisters", "christian support jews"). Use when support targets a faith community as a religious group.
- Women: support for women's rights, empowerment, or wellbeing as a gender group.
- If not Group → "No"
 
Examples:
 
Comment: admired christiano ronaldo leadership good example team follows cant help comment adore
{{"task1": "Supportive", "task2": "Individual", "task3": "No"}}
 
Comment: every word said inspirational inspired legend
{{"task1": "Non-Supportive", "task2": "No", "task3": "No"}}
 
Comment: mashaaallah may allah bless u good health safe long life dear son lot love india fi amaan illah
{{"task1": "Supportive", "task2": "Individual", "task3": "No"}}
 
Comment: messi fan love guy ambition dedication love football
{{"task1": "Supportive", "task2": "Individual", "task3": "No"}}
 
Comment: take nothing man true legend leader messi opinion greatest liverpool fan guy grown right top greatest timered heart
{{"task1": "Non-Supportive", "task2": "No", "task3": "No"}}
 
Comment: new york city support palestine free palestine red heart
{{"task1": "Supportive", "task2": "Group", "task3": "Nation"}}
 
Comment: ya allah help protect people phalestine red heart folded hands crying face
{{"task1": "Supportive", "task2": "Group", "task3": "Nation"}}
 
Comment: cry self saw boy crying oh allah helping syria crying face
{{"task1": "Supportive", "task2": "Group", "task3": "Nation"}}
 
Comment: people country supports israel family supports palestine
{{"task1": "Non-Supportive", "task2": "No", "task3": "No"}}
 
Comment: tear come eyes saw beautiful face forget pain innocent see video palestine babies like
{{"task1": "Non-Supportive", "task2": "No", "task3": "No"}}
 
Comment: heartbreaking grownups create war children pay prayers children lord please protect comfort red heartred heartred heart
{{"task1": "Supportive", "task2": "Group", "task3": "Other"}}
 
Comment: god bless protect children gaza israel innocent victims sickness world
{{"task1": "Supportive", "task2": "Group", "task3": "Other"}}
 
Comment: understand language touching needed encouragement
{{"task1": "Supportive", "task2": "Group", "task3": "Other"}}
 
Comment: support lgbtq community matter sexuality religion anything else still support matter
{{"task1": "Supportive", "task2": "Group", "task3": "LGBTQ"}}
 
Comment: im christian still support lgbtq rights marriage want equality
{{"task1": "Supportive", "task2": "Group", "task3": "LGBTQ"}}
 
Comment: found person huge fan gay boyfriend came help accept afraid turning homophobe
{{"task1": "Non-Supportive", "task2": "No", "task3": "No"}}
 
Comment: racial discrimination support black friends fight get justice raised fistraised fistraised fist
{{"task1": "Supportive", "task2": "Group", "task3": "Black Community"}}
 
Comment: think part like lives matter black lives support black people need justice saying lives matter
{{"task1": "Supportive", "task2": "Group", "task3": "Black Community"}}
 
Comment: blm movement paid racist antiamerican hate group enact division racism within country
{{"task1": "Non-Supportive", "task2": "No", "task3": "No"}}
 
Comment: muslim leaders please help save muslim countries tortured dont close eyes
{{"task1": "Supportive", "task2": "Group", "task3": "Religion"}}
 
Comment: much love muslim palastinian brothers russian eastern orthadox faith
{{"task1": "Supportive", "task2": "Group", "task3": "Religion"}}
 
Comment: jesus loves romans jesus knows ever need
{{"task1": "Non-Supportive", "task2": "No", "task3": "No"}}
 
Comment: moms heros even know
{{"task1": "Supportive", "task2": "Group", "task3": "Women"}}
 
Comment: men care womens success money women understand simple fact
{{"task1": "Non-Supportive", "task2": "No", "task3": "No"}}
 
Comment: resign trump put man charge stop
{{"task1": "Non-Supportive", "task2": "No", "task3": "No"}}
 
Comment: oh god greatest motivation ever heard pretty sure support create life
{{"task1": "Non-Supportive", "task2": "No", "task3": "No"}}
 
Now classify:
Comment: {text}"""

# ── Registry ────────────────────────────────────────────────────────────────

PROMPTS = {
    1: {"zero": TASK1_ZERO_SHOT, "few": TASK1_FEW_SHOT},
    2: {"zero": TASK2_ZERO_SHOT, "few": TASK2_FEW_SHOT},
    3: {"zero": TASK3_ZERO_SHOT, "few": TASK3_FEW_SHOT},
    "combined": {"few": COMBINED_FEW_SHOT},
}


def get_prompt(task, mode: str = "zero") -> str:
    """Get prompt template for a task and mode ('zero' or 'few')."""
    if task not in PROMPTS:
        raise ValueError(f"Unknown task: {task}")
    if mode not in PROMPTS[task]:
        raise ValueError(f"Unknown mode: {mode}. Use 'zero' or 'few'.")
    return PROMPTS[task][mode]
