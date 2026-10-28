"""
Reference information shown next to a prediction: what the condition is, what causes it,
how to recognise it, and how it is usually managed.

Keys are the class folder names the model was trained on (see models/class_names.json).
The advice is general horticultural guidance. Product choice, timing and legal use of
any pesticide depend on the country and crop, so the app always points users to a
local agricultural extension service as well.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiseaseInfo:
    plant: str
    condition: str
    category: str               # Healthy, Fungal, Bacterial, Viral, Oomycete or Pest
    impact: str                 # None, Low, Moderate or High: typical damage if left alone
    pathogen: str = ""          # scientific name of the cause
    summary: str = ""
    symptoms: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)   # what makes it spread
    treatment: list[str] = field(default_factory=list)
    prevention: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return self.category == "Healthy"


# Practices shared by most fungal leaf diseases, appended to their prevention lists
_GOOD_HYGIENE = [
    "Remove and destroy fallen leaves and infected plant debris at the end of the season.",
    "Water at the base of the plant in the morning so leaves dry quickly.",
    "Space and prune plants so air moves freely through the canopy.",
]

_BLIGHT_TOMATO_POTATO = [
    "Rotate away from tomato, potato, pepper and eggplant for at least 2 to 3 years.",
    "Mulch the soil so spores can't splash from the ground onto lower leaves.",
]

DISEASES: dict[str, DiseaseInfo] = {
    # ---------- Apple ----------
    "Apple___Black_rot": DiseaseInfo(
        plant="Apple", condition="Black rot", category="Fungal", impact="Moderate",
        pathogen="Botryosphaeria obtusa",
        summary="A fungus that causes 'frog-eye' leaf spots, a black rot of the fruit and cankers on branches. "
                "It survives in dead wood and shrivelled fruit left on the tree.",
        symptoms=[
            "Small purple flecks on leaves that grow into round brown spots with a purple edge (frog-eye spots).",
            "Spots may have light tan centres; badly affected leaves turn yellow and drop.",
            "Fruit develops a firm brown-to-black rot, often with rings, starting at the blossom end.",
            "Sunken, rough cankers on branches.",
        ],
        conditions=[
            "Warm, wet weather (roughly 20 to 35 °C) from bloom through summer.",
            "Spores come from mummified fruit, dead twigs and cankers on or near the tree.",
        ],
        treatment=[
            "Prune out dead wood and cankers in dry weather, cutting well below visible damage.",
            "Pick off and destroy mummified fruit on the tree and on the ground.",
            "Protect with a labelled fungicide (such as captan or a strobilurin) from bloom until late summer in wet years.",
        ],
        prevention=[
            "Keep trees vigorous and avoid winter injury and sunscald, which give the fungus entry points.",
            "Remove fire-blight strikes and other dead tissue promptly.",
            *_GOOD_HYGIENE[:1],
        ],
    ),
    "Apple___Rust": DiseaseInfo(
        plant="Apple", condition="Cedar apple rust", category="Fungal", impact="Moderate",
        pathogen="Gymnosporangium juniperi-virginianae",
        summary="A rust fungus that needs two hosts: apple and juniper (eastern red cedar). Spores blow from orange galls "
                "on junipers to apple leaves in spring.",
        symptoms=[
            "Bright yellow-orange spots on the upper side of leaves, often with a red border.",
            "Later, small tube-like or cup-shaped structures form on the underside of the spots.",
            "Heavy infection causes early leaf drop and deformed fruit.",
        ],
        conditions=[
            "Wet spring weather between pink bud and about a month after petal fall.",
            "Junipers or red cedars within a few kilometres carrying orange, jelly-like galls after rain.",
        ],
        treatment=[
            "Infected leaves can't be cured; protect new growth with a labelled fungicide (such as myclobutanil or mancozeb) from pink bud to early summer.",
            "Remove galls from nearby junipers in late winter if they are on your property.",
        ],
        prevention=[
            "Plant rust-resistant apple varieties.",
            "Avoid planting junipers and red cedars close to apple trees.",
        ],
    ),
    "Apple___Scab": DiseaseInfo(
        plant="Apple", condition="Apple scab", category="Fungal", impact="Moderate",
        pathogen="Venturia inaequalis",
        summary="The most common apple disease worldwide. The fungus overwinters in fallen leaves and infects new leaves "
                "and fruit during wet spring weather.",
        symptoms=[
            "Olive-green to brown, velvety spots on leaves that later turn dark and may tear.",
            "Leaves can curl, turn yellow and fall early.",
            "Dark, corky, scabby patches on fruit, sometimes with cracking.",
        ],
        conditions=[
            "Cool, wet springs (about 13 to 24 °C) with long periods of leaf wetness.",
            "Spores released from last year's fallen leaves after rain.",
        ],
        treatment=[
            "Apply a labelled fungicide (such as captan, mancozeb or myclobutanil) from green tip until the end of spring rains.",
            "Remove badly infected leaves and fruit to reduce spread later in the season.",
        ],
        prevention=[
            "Rake, shred or compost fallen leaves in autumn to remove the overwintering fungus.",
            "Choose scab-resistant varieties.",
            *_GOOD_HYGIENE[2:],
        ],
    ),

    # ---------- Cherry ----------
    "Cherry___Powdery_mildew": DiseaseInfo(
        plant="Cherry", condition="Powdery mildew", category="Fungal", impact="Moderate",
        pathogen="Podosphaera clandestina",
        summary="A fungus that grows on the surface of young leaves and fruit as a white powder. Unlike most leaf fungi, "
                "it doesn't need rain to infect.",
        symptoms=[
            "White to grey powdery patches on young leaves, usually starting on the underside.",
            "Leaves curl upward, pucker or look distorted.",
            "Fruit may show a faint white bloom and become unmarketable.",
        ],
        conditions=[
            "Warm days with high humidity and cool nights; dense, shaded canopies.",
            "Infections build up after harvest on new shoot growth.",
        ],
        treatment=[
            "Spray sulfur, potassium bicarbonate or a labelled fungicide (such as myclobutanil) at the first signs.",
            "Remove root suckers and water sprouts, which are often infected first.",
        ],
        prevention=[
            "Prune to open the canopy to light and air.",
            "Avoid heavy nitrogen fertiliser, which produces soft, susceptible growth.",
        ],
    ),

    # ---------- Corn ----------
    "Corn___Common_rust": DiseaseInfo(
        plant="Corn (maize)", condition="Common rust", category="Fungal", impact="Low",
        pathogen="Puccinia sorghi",
        summary="A rust fungus whose spores blow in on the wind each year. Most field corn hybrids tolerate it; "
                "sweet corn and susceptible hybrids can lose yield.",
        symptoms=[
            "Small, oval to elongated, cinnamon-brown powdery pustules on both sides of the leaf.",
            "Pustules break open and release rust-coloured spores that rub off on your fingers.",
            "Severe cases turn leaves yellow and dry them out early.",
        ],
        conditions=[
            "Cool temperatures (about 16 to 23 °C) with heavy dew or high humidity.",
            "Spores carried long distances by wind from warmer regions.",
        ],
        treatment=[
            "Usually no treatment is needed on resistant field corn.",
            "On sweet corn or susceptible hybrids, a foliar fungicide (strobilurin or triazole) can be applied when pustules appear early on the upper leaves.",
        ],
        prevention=[
            "Plant hybrids rated resistant to common rust.",
            "Plant early so the crop matures before rust builds up.",
        ],
    ),
    "Corn___Gray_leaf_spot": DiseaseInfo(
        plant="Corn (maize)", condition="Gray leaf spot", category="Fungal", impact="High",
        pathogen="Cercospora zeae-maydis",
        summary="One of the most yield-limiting leaf diseases of corn. The fungus survives in corn residue on the soil "
                "surface, so it is worst in fields with corn after corn and reduced tillage.",
        symptoms=[
            "Narrow, rectangular, tan to grey lesions that run parallel to the leaf veins.",
            "Lesions are sharply bounded by veins and usually 1 to 6 cm long.",
            "Starts on lower leaves and moves up; lesions merge and kill whole leaves.",
        ],
        conditions=[
            "Warm (above 25 °C), humid weather with fog, heavy dew or long periods of leaf wetness.",
            "Infected corn residue left on the soil surface.",
        ],
        treatment=[
            "A foliar fungicide (strobilurin, triazole or a mix) around tasselling is worthwhile when lesions are found on the third leaf below the ear or higher on susceptible hybrids.",
        ],
        prevention=[
            "Plant hybrids with good gray leaf spot resistance.",
            "Rotate to a non-host crop such as soybean for at least one year.",
            "Bury residue with tillage where erosion allows.",
        ],
    ),
    "Corn___Northern_Leaf_Blight": DiseaseInfo(
        plant="Corn (maize)", condition="Northern leaf blight", category="Fungal", impact="Moderate",
        pathogen="Exserohilum turcicum",
        summary="A fungus that produces long, cigar-shaped lesions. Early infection before tasselling can cause "
                "significant yield loss.",
        symptoms=[
            "Long (2.5 to 15 cm), cigar-shaped, grey-green to tan lesions on the leaves.",
            "Lesions are not limited by veins, unlike gray leaf spot, and have smooth edges.",
            "Dark grey spores form on lesions in humid weather.",
        ],
        conditions=[
            "Moderate temperatures (about 18 to 27 °C) with heavy dew and rain.",
            "Infected corn residue on the soil surface.",
        ],
        treatment=[
            "Apply a foliar fungicide between tasselling and silking if lesions are present on the upper leaves of susceptible hybrids.",
        ],
        prevention=[
            "Plant resistant hybrids.",
            "Rotate crops and manage residue to reduce the amount of fungus carried over.",
        ],
    ),

    # ---------- Grape ----------
    "Grape___Black_rot": DiseaseInfo(
        plant="Grape", condition="Black rot", category="Fungal", impact="High",
        pathogen="Guignardia bidwellii",
        summary="A fungus that spots leaves and turns berries into hard, black, shrivelled mummies. In wet years it can "
                "destroy most of the crop.",
        symptoms=[
            "Round, tan to reddish-brown leaf spots with a dark border.",
            "Tiny black dots (fungal fruiting bodies) inside the spots, often in a ring.",
            "Berries turn brown, then black, shrink and stay attached as mummies.",
        ],
        conditions=[
            "Warm, rainy weather from bud break until berries begin to ripen.",
            "Mummified berries and infected canes left in the vineyard.",
        ],
        treatment=[
            "Apply a labelled fungicide (such as mancozeb, captan or myclobutanil) from early shoot growth until about four weeks after bloom.",
            "Remove and destroy mummies and infected clusters.",
        ],
        prevention=[
            "Remove all mummies from vines and the ground before bud break.",
            "Train and prune vines for good air flow and sun exposure.",
        ],
    ),
    "Grape___Esca": DiseaseInfo(
        plant="Grape", condition="Esca (black measles)", category="Fungal", impact="High",
        pathogen="Phaeomoniella chlamydospora, Phaeoacremonium spp. and others",
        summary="A trunk disease caused by several wood-rotting fungi that enter through pruning wounds. It develops "
                "over years and can kill vines suddenly.",
        symptoms=[
            "'Tiger stripe' leaves: yellow or red bands between the veins that dry to brown.",
            "Small dark spots on berries (black measles); berries may crack and dry.",
            "Dark streaks or spots in the wood when a cane or trunk is cut.",
            "Sudden wilting and death of a whole vine in hot weather (apoplexy).",
        ],
        conditions=[
            "Large pruning wounds made in wet weather, when spores are spreading.",
            "Older vines and vines under drought or heat stress show symptoms most.",
        ],
        treatment=[
            "There is no cure once the wood is infected.",
            "Cut out affected arms or trunks well below visible staining and retrain a new trunk from a healthy shoot.",
            "Remove and burn dead vines and pruned wood.",
        ],
        prevention=[
            "Prune late in the dormant season, in dry weather, and avoid very large cuts.",
            "Protect fresh pruning wounds with a wound sealant or labelled product.",
            "Plant certified clean nursery stock.",
        ],
    ),
    "Grape___Leaf_blight": DiseaseInfo(
        plant="Grape", condition="Leaf blight (Isariopsis leaf spot)", category="Fungal", impact="Low",
        pathogen="Pseudocercospora vitis",
        summary="A leaf-spotting fungus that appears later in the season. It mainly causes early leaf drop, which can "
                "weaken vines over time.",
        symptoms=[
            "Irregular dark red-brown to black spots on older leaves.",
            "Spots enlarge and merge, and the tissue dries out.",
            "Leaves yellow and fall early.",
        ],
        conditions=[
            "Warm, humid weather in mid to late summer.",
            "Infected leaves left on the vineyard floor.",
        ],
        treatment=[
            "Fungicide programmes used for other grape diseases (such as mancozeb or copper) usually control it.",
            "Remove badly spotted leaves.",
        ],
        prevention=[*_GOOD_HYGIENE],
    ),

    # ---------- Orange ----------
    "Orange": DiseaseInfo(
        plant="Orange", condition="Citrus greening (Huanglongbing)", category="Bacterial", impact="High",
        pathogen="Candidatus Liberibacter asiaticus",
        summary="The most destructive citrus disease in the world. A bacterium living in the tree's sap, spread by the "
                "tiny Asian citrus psyllid insect and by infected grafting material. Infected trees can't be cured.",
        symptoms=[
            "Blotchy, uneven yellowing of leaves that is not the same on both halves of the leaf.",
            "Yellow shoots, small upright leaves and twig dieback.",
            "Small, lopsided, bitter fruit that stays partly green; fruit drops early.",
        ],
        conditions=[
            "Asian citrus psyllids feeding on new leaf flushes.",
            "Planting or grafting with infected budwood or nursery trees.",
        ],
        treatment=[
            "There is no cure. Remove and destroy confirmed infected trees to protect healthy ones.",
            "Control psyllids with labelled insecticides, especially during new growth flushes.",
            "Good nutrition and irrigation can slow decline but won't remove the infection.",
        ],
        prevention=[
            "Buy only certified disease-free nursery trees.",
            "Inspect new growth often for psyllids.",
            "Citrus greening is a regulated disease in many countries. Report suspected cases to your local agricultural authority.",
        ],
    ),

    # ---------- Peach ----------
    "Peach___Bacterial_spot": DiseaseInfo(
        plant="Peach", condition="Bacterial spot", category="Bacterial", impact="Moderate",
        pathogen="Xanthomonas arboricola pv. pruni",
        summary="A bacterium that spots leaves and fruit. It overwinters in twig cankers and spreads in wind-driven rain.",
        symptoms=[
            "Small, angular, water-soaked spots that turn purple-brown to black.",
            "Spot centres fall out, leaving a 'shot-hole' look.",
            "Leaves turn yellow and drop; fruit gets pitted or cracked spots.",
        ],
        conditions=[
            "Warm, wet weather with frequent wind and rain.",
            "Light, sandy soils and stressed trees are more affected.",
        ],
        treatment=[
            "Copper sprays from leaf fall through early spring reduce overwintering bacteria.",
            "During the season, labelled bactericides (such as oxytetracycline) give partial control.",
            "Control is limited once it is established; focus on protecting new growth.",
        ],
        prevention=[
            "Plant resistant varieties; this is the most effective control.",
            "Use windbreaks and keep trees well fed and watered.",
        ],
    ),

    # ---------- Pepper ----------
    "Pepper___Bacterial_spot": DiseaseInfo(
        plant="Bell pepper", condition="Bacterial spot", category="Bacterial", impact="High",
        pathogen="Xanthomonas euvesicatoria and related species",
        summary="A seed-borne bacterium that spreads quickly in warm, wet weather and can strip plants of their leaves.",
        symptoms=[
            "Small, water-soaked spots that turn brown, sometimes with a yellow halo.",
            "Spots merge; leaves yellow and drop, exposing fruit to sunscald.",
            "Raised, scabby, brown spots on the fruit.",
        ],
        conditions=[
            "Warm temperatures (24 to 30 °C) with rain, overhead watering or high humidity.",
            "Infected seed, transplants or volunteer plants.",
        ],
        treatment=[
            "Remove heavily infected plants.",
            "Spray copper combined with mancozeb to slow spread; copper alone may fail where the bacteria are resistant.",
            "Avoid working with plants while they are wet.",
        ],
        prevention=[
            "Use certified disease-free seed and transplants, and resistant varieties.",
            "Rotate away from pepper and tomato for 2 to 3 years.",
            "Use drip irrigation instead of sprinklers.",
        ],
    ),

    # ---------- Potato ----------
    "Potato___Early_blight": DiseaseInfo(
        plant="Potato", condition="Early blight", category="Fungal", impact="Moderate",
        pathogen="Alternaria solani",
        summary="A common fungus that attacks older, lower leaves first. It is worst on plants that are stressed or "
                "short of nitrogen.",
        symptoms=[
            "Dark brown spots with rings inside them, like a target.",
            "A yellow halo often surrounds each spot.",
            "Starts on the oldest leaves and moves upward; leaves die and drop.",
        ],
        conditions=[
            "Warm weather (about 24 to 29 °C) with alternating wet and dry periods.",
            "Plant stress from drought, poor nutrition or heavy fruit or tuber load.",
        ],
        treatment=[
            "Remove infected lower leaves.",
            "Apply a labelled fungicide (such as chlorothalonil, mancozeb or azoxystrobin) when spots first appear, and repeat as the label allows.",
        ],
        prevention=[
            "Keep plants well fed, especially with nitrogen, and evenly watered.",
            *_BLIGHT_TOMATO_POTATO,
            *_GOOD_HYGIENE[:1],
        ],
    ),
    "Potato___Late_blight": DiseaseInfo(
        plant="Potato", condition="Late blight", category="Oomycete", impact="High",
        pathogen="Phytophthora infestans",
        summary="The water mould behind the Irish potato famine. In cool, wet weather it can destroy a whole field in "
                "about a week and spreads easily to neighbouring potato and tomato plants.",
        symptoms=[
            "Large, pale green to dark brown, water-soaked patches, often at leaf tips and edges.",
            "White, fuzzy growth on the underside of leaves in humid conditions.",
            "Dark lesions on stems; tubers develop a reddish-brown, dry rot under the skin.",
        ],
        conditions=[
            "Cool (10 to 24 °C), wet weather with high humidity or fog.",
            "Infected seed tubers, cull piles and volunteer potatoes.",
        ],
        treatment=[
            "Act quickly. Remove and bag infected plants; don't compost them.",
            "Protect nearby plants with a labelled fungicide (such as chlorothalonil or mancozeb, or late-blight specific products).",
            "Kill the vines two to three weeks before harvest so spores don't reach the tubers.",
        ],
        prevention=[
            "Plant certified disease-free seed potatoes and resistant varieties.",
            "Destroy cull piles and volunteer potato plants.",
            "Hill soil up around plants to protect tubers.",
        ],
    ),

    # ---------- Squash ----------
    "Squash": DiseaseInfo(
        plant="Squash", condition="Powdery mildew", category="Fungal", impact="Moderate",
        pathogen="Podosphaera xanthii",
        summary="A very common fungus on squash, pumpkin and cucumber. It covers leaves with white powder and reduces "
                "the plant's ability to make food, which lowers yield and fruit quality.",
        symptoms=[
            "White, powdery spots on both sides of the leaves and on stems.",
            "Spots spread until the whole leaf is covered.",
            "Leaves turn yellow, then brown and dry.",
        ],
        conditions=[
            "Warm, dry days and humid nights; it doesn't need rain to spread.",
            "Shaded, crowded plantings.",
        ],
        treatment=[
            "Spray sulfur, potassium bicarbonate, neem oil or horticultural oil at the first spots.",
            "Rotate between fungicide groups if you use synthetic fungicides, because this fungus becomes resistant quickly.",
            "Remove the worst-affected leaves.",
        ],
        prevention=[
            "Plant mildew-resistant varieties.",
            "Grow in full sun and give plants enough space.",
        ],
    ),

    # ---------- Strawberry ----------
    "Strawberry___Leaf_scorch": DiseaseInfo(
        plant="Strawberry", condition="Leaf scorch", category="Fungal", impact="Moderate",
        pathogen="Diplocarpon earlianum",
        summary="A leaf fungus that makes strawberry leaves look burned. Heavy infection weakens plants and reduces "
                "the next year's crop.",
        symptoms=[
            "Many small, irregular, dark purple blotches on the upper side of leaves.",
            "Unlike leaf spot, the blotches don't develop white centres.",
            "Blotches merge and the leaf dries out and curls, looking scorched.",
        ],
        conditions=[
            "Warm, wet weather and long periods of leaf wetness.",
            "Old, crowded beds where infected leaves build up.",
        ],
        treatment=[
            "Remove and destroy infected leaves.",
            "Renovate beds right after harvest by mowing old leaves and thinning plants.",
            "Apply a labelled fungicide (such as captan) in spring if the disease was bad the year before.",
        ],
        prevention=[
            "Plant resistant varieties and certified disease-free plants.",
            "Use drip irrigation and replace beds every few years.",
        ],
    ),

    # ---------- Tomato ----------
    "Tomato___Bacterial_spot": DiseaseInfo(
        plant="Tomato", condition="Bacterial spot", category="Bacterial", impact="High",
        pathogen="Xanthomonas spp.",
        summary="A seed-borne bacterium that spots leaves, stems and fruit and spreads quickly in warm, wet weather.",
        symptoms=[
            "Small (under 3 mm), dark, water-soaked spots on leaves, sometimes with a yellow halo.",
            "Spots merge and leaves turn yellow and drop.",
            "Raised, scabby spots on green fruit.",
        ],
        conditions=[
            "Warm temperatures (24 to 30 °C) with rain, splashing water or high humidity.",
            "Infected seed, transplants or plant debris.",
        ],
        treatment=[
            "Remove heavily infected plants.",
            "Spray copper combined with mancozeb to slow spread.",
            "Avoid handling plants when they are wet.",
        ],
        prevention=[
            "Use certified disease-free seed and transplants.",
            "Rotate away from tomato and pepper for 2 to 3 years.",
            "Use drip irrigation and mulch.",
        ],
    ),
    "Tomato___Early_blight": DiseaseInfo(
        plant="Tomato", condition="Early blight", category="Fungal", impact="Moderate",
        pathogen="Alternaria solani / A. linariae",
        summary="One of the most common tomato diseases. It starts on old, lower leaves and works its way up the plant.",
        symptoms=[
            "Brown spots with rings inside them, like a target, often with a yellow halo.",
            "Lower leaves turn yellow, die and drop.",
            "Dark, sunken lesions can form on stems and at the stem end of fruit.",
        ],
        conditions=[
            "Warm weather (about 24 to 29 °C) with heavy dew or frequent rain.",
            "Spores in the soil and on last season's plant debris.",
        ],
        treatment=[
            "Remove infected lower leaves.",
            "Apply a labelled fungicide (such as chlorothalonil, mancozeb, copper or azoxystrobin) when spots first appear.",
        ],
        prevention=[
            "Stake or cage plants and remove the lowest leaves so they don't touch the soil.",
            *_BLIGHT_TOMATO_POTATO,
            *_GOOD_HYGIENE[1:2],
        ],
    ),
    "Tomato___Late_blight": DiseaseInfo(
        plant="Tomato", condition="Late blight", category="Oomycete", impact="High",
        pathogen="Phytophthora infestans",
        summary="A fast-spreading water mould that can kill tomato plants within days in cool, wet weather. "
                "The same organism causes late blight in potatoes.",
        symptoms=[
            "Large, greasy-looking, grey-green to brown patches on leaves.",
            "White, fuzzy growth on the underside of leaves in humid weather.",
            "Dark brown lesions on stems; firm, brown, greasy patches on fruit.",
        ],
        conditions=[
            "Cool (10 to 24 °C), wet or foggy weather.",
            "Spores blowing in from infected tomato or potato plants nearby.",
        ],
        treatment=[
            "Act quickly. Remove and bag infected plants; don't compost them.",
            "Protect healthy plants with a labelled fungicide (such as chlorothalonil, mancozeb or copper).",
        ],
        prevention=[
            "Plant resistant varieties.",
            "Don't grow tomatoes next to potatoes, and destroy volunteer potato plants.",
            *_GOOD_HYGIENE[1:],
        ],
    ),
    "Tomato___Leaf_Mold": DiseaseInfo(
        plant="Tomato", condition="Leaf mould", category="Fungal", impact="Moderate",
        pathogen="Passalora fulva (Cladosporium fulvum)",
        summary="A fungus that thrives in very humid air, so it is mostly a problem in greenhouses and polytunnels.",
        symptoms=[
            "Pale green or yellow spots on the upper side of leaves.",
            "Olive-green to greyish-purple velvety mould directly underneath the spots.",
            "Leaves curl, wither and drop; fruit is rarely affected.",
        ],
        conditions=[
            "Relative humidity above 85% with moderate temperatures.",
            "Poor ventilation in greenhouses.",
        ],
        treatment=[
            "Increase ventilation and lower humidity straight away.",
            "Remove infected leaves.",
            "Apply a labelled fungicide (such as chlorothalonil or copper) if it keeps spreading.",
        ],
        prevention=[
            "Grow resistant varieties.",
            "Vent and heat greenhouses to keep leaves dry, and avoid wetting foliage.",
            "Clean the greenhouse thoroughly between crops.",
        ],
    ),
    "Tomato___Mosaic_virus": DiseaseInfo(
        plant="Tomato", condition="Tomato mosaic virus", category="Viral", impact="Moderate",
        pathogen="Tomato mosaic virus (ToMV, Tobamovirus)",
        summary="A very stable virus spread by touch: on hands, tools, clothing and seed. It can survive for years in "
                "plant debris. Infected plants can't be cured.",
        symptoms=[
            "Light and dark green mottled (mosaic) pattern on leaves.",
            "Leaves curled, wrinkled or narrow and fern-like.",
            "Stunted plants, fewer fruit, and uneven ripening.",
        ],
        conditions=[
            "Handling healthy plants after touching infected ones.",
            "Infected seed or tobacco products used near plants.",
        ],
        treatment=[
            "There is no cure. Remove and destroy infected plants, roots included.",
            "Disinfect tools and wash hands with soap after handling suspect plants.",
        ],
        prevention=[
            "Grow resistant varieties (look for ToMV or Tm-2 resistance on the seed packet).",
            "Use certified disease-free seed.",
            "Don't smoke or use tobacco products while working with plants.",
        ],
    ),
    "Tomato___Septoria_leaf_spot": DiseaseInfo(
        plant="Tomato", condition="Septoria leaf spot", category="Fungal", impact="Moderate",
        pathogen="Septoria lycopersici",
        summary="A fungus that covers lower leaves with many small spots. It rarely affects fruit but can strip the plant "
                "of leaves, leading to sunscald and lower yield.",
        symptoms=[
            "Many small (1.5 to 6 mm), round spots with dark edges and grey or tan centres.",
            "Tiny black dots (fungal fruiting bodies) in the centre of the spots.",
            "Starts on the lowest leaves after fruit sets; leaves yellow and drop.",
        ],
        conditions=[
            "Warm, wet weather (20 to 25 °C) and splashing water.",
            "Infected tomato debris and weeds in the nightshade family.",
        ],
        treatment=[
            "Remove infected lower leaves.",
            "Apply a labelled fungicide (such as chlorothalonil, mancozeb or copper) and repeat as the label allows.",
        ],
        prevention=[
            *_BLIGHT_TOMATO_POTATO,
            "Control nightshade weeds around the garden.",
            *_GOOD_HYGIENE[1:2],
        ],
    ),
    "Tomato___Spider_mites": DiseaseInfo(
        plant="Tomato", condition="Spider mites", category="Pest", impact="Moderate",
        pathogen="Tetranychus urticae (two-spotted spider mite)",
        summary="Not a disease but tiny sap-sucking mites that live on the underside of leaves. Populations explode in "
                "hot, dry weather.",
        symptoms=[
            "Fine yellow or white speckles (stippling) on the upper side of leaves.",
            "Leaves turn bronze or yellow and dry out.",
            "Fine webbing and tiny moving dots on the underside of leaves.",
        ],
        conditions=[
            "Hot, dry, dusty conditions.",
            "Broad-spectrum insecticides that kill the mites' natural predators.",
        ],
        treatment=[
            "Spray the undersides of leaves with a strong jet of water to knock mites off.",
            "Use insecticidal soap or horticultural oil, covering the underside of the leaves.",
            "For heavy infestations, use a labelled miticide (such as abamectin or bifenazate), or release predatory mites.",
        ],
        prevention=[
            "Keep plants well watered and reduce dust.",
            "Avoid broad-spectrum insecticides so predators survive.",
            "Check the underside of leaves regularly in hot weather.",
        ],
    ),
    "Tomato___Target_Spot": DiseaseInfo(
        plant="Tomato", condition="Target spot", category="Fungal", impact="Moderate",
        pathogen="Corynespora cassiicola",
        summary="A fungus that causes ringed spots on leaves, stems and fruit, mainly in warm, humid regions.",
        symptoms=[
            "Brown spots with light centres and rings, often with a yellow halo.",
            "Spots are usually smaller than early blight and may crack in the centre.",
            "Sunken, ringed spots on fruit; leaves drop when infection is heavy.",
        ],
        conditions=[
            "Warm (20 to 28 °C), humid weather with long periods of leaf wetness.",
            "Dense canopies and leftover crop debris.",
        ],
        treatment=[
            "Remove infected leaves and improve air flow.",
            "Apply a labelled fungicide (such as chlorothalonil or azoxystrobin) when spots first appear.",
        ],
        prevention=[*_GOOD_HYGIENE, "Rotate crops and avoid excessive nitrogen."],
    ),
    "Tomato___Yellow_Leaf_Curl_Virus": DiseaseInfo(
        plant="Tomato", condition="Tomato yellow leaf curl virus", category="Viral", impact="High",
        pathogen="Tomato yellow leaf curl virus (TYLCV, Begomovirus)",
        summary="A virus carried by the silverleaf whitefly. Plants infected young may produce almost no fruit. "
                "There is no cure, so control focuses on whiteflies.",
        symptoms=[
            "Leaves curl upward and inward and are small and thick.",
            "Yellowing at the leaf edges and between the veins.",
            "Stunted, bushy plants; flowers drop and few fruit set.",
        ],
        conditions=[
            "Whiteflies (Bemisia tabaci) carrying the virus from infected plants or weeds.",
            "Warm climates and greenhouses where whiteflies thrive.",
        ],
        treatment=[
            "There is no cure. Remove infected plants and bag them to avoid spreading whiteflies.",
            "Control whiteflies with yellow sticky traps, insecticidal soap or labelled insecticides.",
        ],
        prevention=[
            "Plant TYLCV-resistant varieties.",
            "Protect seedlings with fine insect netting and use reflective mulch.",
            "Remove weeds that host whiteflies.",
        ],
    ),
}

# Plant names for the healthy classes, including the single-class healthy plants
_HEALTHY_PLANTS = {
    "Apple___healthy": "Apple",
    "Blueberry": "Blueberry",
    "Cherry___healthy": "Cherry",
    "Corn___healthy": "Corn (maize)",
    "Grape___healthy": "Grape",
    "Peach___healthy": "Peach",
    "Pepper___healthy": "Bell pepper",
    "Potato___healthy": "Potato",
    "Raspberry": "Raspberry",
    "Soybean": "Soybean",
    "Strawberry___healthy": "Strawberry",
    "Tomato___healthy": "Tomato",
}

# These plants only have one class in PlantVillage, so the model knows the plant, not its condition.
SINGLE_CLASS_PLANTS = {"Blueberry", "Orange", "Raspberry", "Soybean", "Squash"}

for _name, _plant in _HEALTHY_PLANTS.items():
    DISEASES[_name] = DiseaseInfo(
        plant=_plant, condition="Healthy", category="Healthy", impact="None",
        summary=f"The leaf matches healthy {_plant.lower()} leaves the model saw in training. "
                "No signs of the diseases it knows were found.",
        prevention=[
            "Keep checking leaves weekly, including the underside, especially after warm, wet spells.",
            "Water at the base of the plant and avoid wetting the leaves.",
            "Remove fallen leaves and plant debris that can carry disease into next season.",
            "Feed and water plants evenly; stressed plants get sick more easily.",
        ],
    )


def get_info(class_name: str) -> DiseaseInfo:
    """Reference info for a class folder name, with a plain fallback for unknown classes."""
    if class_name in DISEASES:
        return DISEASES[class_name]
    plant, _, condition = class_name.partition("___")
    condition = condition.replace("_", " ") or "Unknown"
    healthy = condition.lower() == "healthy"
    return DiseaseInfo(
        plant=plant.replace("_", " "), condition=condition.capitalize(),
        category="Healthy" if healthy else "Unknown", impact="None" if healthy else "Unknown",
    )


def related_conditions(class_name: str, class_names: list[str]) -> list[str]:
    """Other classes for the same plant, so a 'healthy' result can list what to watch for."""
    plant = class_name.partition("___")[0]
    return [n for n in class_names if n != class_name and n.partition("___")[0] == plant]
