import json
import random
import os
import matplotlib.pyplot as plt
import argparse
import json
import pandas as pd
from typing import Optional
from robustbench.data import load_cifar10, load_cifar100, load_cifar10c, load_cifar100c, load_imagenet, load_imagenet3dcc
from torchvision import datasets


def load_dataset(name, n_examples, corruption=None, severity=5):

    if name == 'cifar10':
        x_test, y_test = load_cifar10(n_examples=n_examples, data_dir="../../data/RobustBench/data/")
    elif name == 'cifar100':
        x_test, y_test = load_cifar100(n_examples=n_examples, data_dir="../../data/RobustBench/data/")
    elif name == 'imagenet':
        x_test, y_test = load_imagenet(n_examples=n_examples, data_dir="../../data/RobustBench/data/")
    elif name == 'cifar10c':
        if not corruption: raise ValueError("cifar10c requires specifying a corruption")
        x_test, y_test = load_cifar10c(n_examples=n_examples, corruptions=[corruption], severity=severity, data_dir="../../data/RobustBench/data/")
    elif name == 'cifar100c':
        if not corruption: raise ValueError("cifar100c requires specifying a corruption")
        x_test, y_test = load_cifar100c(n_examples=n_examples, corruptions=[corruption], severity=severity, data_dir="../../data/RobustBench/data/")
    elif name == 'imagenet3dcc':
        if not corruption: raise ValueError("imagenet3dcc requires specifying a corruption")
        x_test, y_test = load_imagenet3dcc(n_examples=n_examples, corruptions=[corruption], severity=severity, data_dir="../../data/RobustBench/data/")
    else:
        raise ValueError(f"unsupported dataset {name}")
    return x_test, y_test

def visualize_positives_from_json(json_path, max_samples):
    # Parse the filename to determine dataset and corruption details
    filename = os.path.basename(json_path)
    corruption = None
    severity = None
    
    model = "-".join(filename.split('-')[1:-4])
    if filename.startswith('cifar10c-'):
        dataset_clean = 'cifar10'
        dataset_corrupt = 'cifar10c'
        threshold = 10000
        corruption = filename.split('-')[-2]
        severity = int(filename.split('-')[-1].split('.')[0])
    elif filename.startswith('cifar100c-'):
        dataset_clean = 'cifar100'
        dataset_corrupt = 'cifar100c'
        threshold = 10000
        corruption = filename.split('-')[-2]
        severity = int(filename.split('-')[-1].split('.')[0])
    elif filename.startswith('imagenet3dcc-'):
        dataset_clean = 'imagenet'
        dataset_corrupt = 'imagenet3dcc'
        threshold = 5000
        corruption = filename.split('-')[-2]
        severity = int(filename.split('-')[-1].split('.')[0])
    else:
        raise ValueError(f"Unsupported dataset in filename: {filename}")

    # Load positives from JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    positives = data['positives']
    np = len(positives)
    perc = len(set([t[1] for t in positives])) / (2*threshold)
    
    # Sample if too many positives
    if len(positives) > max_samples:
        positives = random.sample(positives, max_samples)
    
    # Load datasets
    x_clean, _ = load_dataset(dataset_clean, n_examples=threshold)
    x_corrupt, _ = load_dataset(dataset_corrupt, n_examples=threshold, corruption=corruption, severity=severity)

    # Load model predictions
    pred_clean = pd.read_csv(f"../../data/RobustBench/predictions/{dataset_corrupt}-{model}/vanilla.csv")["pred"]
    pred_corrupt = pd.read_csv(f"../../data/RobustBench/predictions/{dataset_corrupt}-{model}/{corruption}-{severity}.csv")["pred"]

    # Map numerical predictions to class names
    if dataset_clean == 'cifar10':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_clean == 'cifar100':  # cifar100
        class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    else:
        class_names = ["tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead", "electric_ray", "stingray", "cock", "hen", "ostrich", "brambling", "goldfinch", "house_finch", "junco", "indigo_bunting", "robin", "bulbul", "jay", "magpie", "chickadee", "water_ouzel", "kite", "bald_eagle", "vulture", "great_grey_owl", "European_fire_salamander", "common_newt", "eft", "spotted_salamander", "axolotl", "bullfrog", "tree_frog", "tailed_frog", "loggerhead", "leatherback_turtle", "mud_turtle", "terrapin", "box_turtle", "banded_gecko", "common_iguana", "American_chameleon", "whiptail", "agama", "frilled_lizard", "alligator_lizard", "Gila_monster", "green_lizard", "African_chameleon", "Komodo_dragon", "African_crocodile", "American_alligator", "triceratops", "thunder_snake", "ringneck_snake", "hognose_snake", "green_snake", "king_snake", "garter_snake", "water_snake", "vine_snake", "night_snake", "boa_constrictor", "rock_python", "Indian_cobra", "green_mamba", "sea_snake", "horned_viper", "diamondback", "sidewinder", "trilobite", "harvestman", "scorpion", "black_and_gold_garden_spider", "barn_spider", "garden_spider", "black_widow", "tarantula", "wolf_spider", "tick", "centipede", "black_grouse", "ptarmigan", "ruffed_grouse", "prairie_chicken", "peacock", "quail", "partridge", "African_grey", "macaw", "sulphur-crested_cockatoo", "lorikeet", "coucal", "bee_eater", "hornbill", "hummingbird", "jacamar", "toucan", "drake", "red-breasted_merganser", "goose", "black_swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea_anemone", "brain_coral", "flatworm", "nematode", "conch", "snail", "slug", "sea_slug", "chiton", "chambered_nautilus", "Dungeness_crab", "rock_crab", "fiddler_crab", "king_crab", "American_lobster", "spiny_lobster", "crayfish", "hermit_crab", "isopod", "white_stork", "black_stork", "spoonbill", "flamingo", "little_blue_heron", "American_egret", "bittern", "crane", "limpkin", "European_gallinule", "American_coot", "bustard", "ruddy_turnstone", "red-backed_sandpiper", "redshank", "dowitcher", "oystercatcher", "pelican", "king_penguin", "albatross", "grey_whale", "killer_whale", "dugong", "sea_lion", "Chihuahua", "Japanese_spaniel", "Maltese_dog", "Pekinese", "Shih-Tzu", "Blenheim_spaniel", "papillon", "toy_terrier", "Rhodesian_ridgeback", "Afghan_hound", "basset", "beagle", "bloodhound", "bluetick", "black-and-tan_coonhound", "Walker_hound", "English_foxhound", "redbone", "borzoi", "Irish_wolfhound", "Italian_greyhound", "whippet", "Ibizan_hound", "Norwegian_elkhound", "otterhound", "Saluki", "Scottish_deerhound", "Weimaraner", "Staffordshire_bullterrier", "American_Staffordshire_terrier", "Bedlington_terrier", "Border_terrier", "Kerry_blue_terrier", "Irish_terrier", "Norfolk_terrier", "Norwich_terrier", "Yorkshire_terrier", "wire-haired_fox_terrier", "Lakeland_terrier", "Sealyham_terrier", "Airedale", "cairn", "Australian_terrier", "Dandie_Dinmont", "Boston_bull", "miniature_schnauzer", "giant_schnauzer", "standard_schnauzer", "Scotch_terrier", "Tibetan_terrier", "silky_terrier", "soft-coated_wheaten_terrier", "West_Highland_white_terrier", "Lhasa", "flat-coated_retriever", "curly-coated_retriever", "golden_retriever", "Labrador_retriever", "Chesapeake_Bay_retriever", "German_short-haired_pointer", "vizsla", "English_setter", "Irish_setter", "Gordon_setter", "Brittany_spaniel", "clumber", "English_springer", "Welsh_springer_spaniel", "cocker_spaniel", "Sussex_spaniel", "Irish_water_spaniel", "kuvasz", "schipperke", "groenendael", "malinois", "briard", "kelpie", "komondor", "Old_English_sheepdog", "Shetland_sheepdog", "collie", "Border_collie", "Bouvier_des_Flandres", "Rottweiler", "German_shepherd", "Doberman", "miniature_pinscher", "Greater_Swiss_Mountain_dog", "Bernese_mountain_dog", "Appenzeller", "EntleBucher", "boxer", "bull_mastiff", "Tibetan_mastiff", "French_bulldog", "Great_Dane", "Saint_Bernard", "Eskimo_dog", "malamute", "Siberian_husky", "dalmatian", "affenpinscher", "basenji", "pug", "Leonberg", "Newfoundland", "Great_Pyrenees", "Samoyed", "Pomeranian", "chow", "keeshond", "Brabancon_griffon", "Pembroke", "Cardigan", "toy_poodle", "miniature_poodle", "standard_poodle", "Mexican_hairless", "timber_wolf", "white_wolf", "red_wolf", "coyote", "dingo", "dhole", "African_hunting_dog", "hyena", "red_fox", "kit_fox", "Arctic_fox", "grey_fox", "tabby", "tiger_cat", "Persian_cat", "Siamese_cat", "Egyptian_cat", "cougar", "lynx", "leopard", "snow_leopard", "jaguar", "lion", "tiger", "cheetah", "brown_bear", "American_black_bear", "ice_bear", "sloth_bear", "mongoose", "meerkat", "tiger_beetle", "ladybug", "ground_beetle", "long-horned_beetle", "leaf_beetle", "dung_beetle", "rhinoceros_beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket", "walking_stick", "cockroach", "mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "admiral", "ringlet", "monarch", "cabbage_butterfly", "sulphur_butterfly", "lycaenid", "starfish", "sea_urchin", "sea_cucumber", "wood_rabbit", "hare", "Angora", "hamster", "porcupine", "fox_squirrel", "marmot", "beaver", "guinea_pig", "sorrel", "zebra", "hog", "wild_boar", "warthog", "hippopotamus", "ox", "water_buffalo", "bison", "ram", "bighorn", "ibex", "hartebeest", "impala", "gazelle", "Arabian_camel", "llama", "weasel", "mink", "polecat", "black-footed_ferret", "otter", "skunk", "badger", "armadillo", "three-toed_sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas", "baboon", "macaque", "langur", "colobus", "proboscis_monkey", "marmoset", "capuchin", "howler_monkey", "titi", "spider_monkey", "squirrel_monkey", "Madagascar_cat", "indri", "Indian_elephant", "African_elephant", "lesser_panda", "giant_panda", "barracouta", "eel", "coho", "rock_beauty", "anemone_fish", "sturgeon", "gar", "lionfish", "puffer", "abacus", "abaya", "academic_gown", "accordion", "acoustic_guitar", "aircraft_carrier", "airliner", "airship", "altar", "ambulance", "amphibian", "analog_clock", "apiary", "apron", "ashcan", "assault_rifle", "backpack", "bakery", "balance_beam", "balloon", "ballpoint", "Band_Aid", "banjo", "bannister", "barbell", "barber_chair", "barbershop", "barn", "barometer", "barrel", "barrow", "baseball", "basketball", "bassinet", "bassoon", "bathing_cap", "bath_towel", "bathtub", "beach_wagon", "beacon", "beaker", "bearskin", "beer_bottle", "beer_glass", "bell_cote", "bib", "bicycle-built-for-two", "bikini", "binder", "binoculars", "birdhouse", "boathouse", "bobsled", "bolo_tie", "bonnet", "bookcase", "bookshop", "bottlecap", "bow", "bow_tie", "brass", "brassiere", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof_vest", "bullet_train", "butcher_shop", "cab", "caldron", "candle", "cannon", "canoe", "can_opener", "cardigan", "car_mirror", "carousel", "carpenter's_kit", "carton", "car_wheel", "cash_machine", "cassette", "cassette_player", "castle", "catamaran", "CD_player", "cello", "cellular_telephone", "chain", "chainlink_fence", "chain_mail", "chain_saw", "chest", "chiffonier", "chime", "china_cabinet", "Christmas_stocking", "church", "cinema", "cleaver", "cliff_dwelling", "cloak", "clog", "cocktail_shaker", "coffee_mug", "coffeepot", "coil", "combination_lock", "computer_keyboard", "confectionery", "container_ship", "convertible", "corkscrew", "cornet", "cowboy_boot", "cowboy_hat", "cradle", "crane", "crash_helmet", "crate", "crib", "Crock_Pot", "croquet_ball", "crutch", "cuirass", "dam", "desk", "desktop_computer", "dial_telephone", "diaper", "digital_clock", "digital_watch", "dining_table", "dishrag", "dishwasher", "disk_brake", "dock", "dogsled", "dome", "doormat", "drilling_platform", "drum", "drumstick", "dumbbell", "Dutch_oven", "electric_fan", "electric_guitar", "electric_locomotive", "entertainment_center", "envelope", "espresso_maker", "face_powder", "feather_boa", "file", "fireboat", "fire_engine", "fire_screen", "flagpole", "flute", "folding_chair", "football_helmet", "forklift", "fountain", "fountain_pen", "four-poster", "freight_car", "French_horn", "frying_pan", "fur_coat", "garbage_truck", "gasmask", "gas_pump", "goblet", "go-kart", "golf_ball", "golfcart", "gondola", "gong", "gown", "grand_piano", "greenhouse", "grille", "grocery_store", "guillotine", "hair_slide", "hair_spray", "half_track", "hammer", "hamper", "hand_blower", "hand-held_computer", "handkerchief", "hard_disc", "harmonica", "harp", "harvester", "hatchet", "holster", "home_theater", "honeycomb", "hook", "hoopskirt", "horizontal_bar", "horse_cart", "hourglass", "iPod", "iron", "jack-o'-lantern", "jean", "jeep", "jersey", "jigsaw_puzzle", "jinrikisha", "joystick", "kimono", "knee_pad", "knot", "lab_coat", "ladle", "lampshade", "laptop", "lawn_mower", "lens_cap", "letter_opener", "library", "lifeboat", "lighter", "limousine", "liner", "lipstick", "Loafer", "lotion", "loudspeaker", "loupe", "lumbermill", "magnetic_compass", "mailbag", "mailbox", "maillot", "maillot", "manhole_cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring_cup", "medicine_chest", "megalith", "microphone", "microwave", "military_uniform", "milk_can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing_bowl", "mobile_home", "Model_T", "modem", "monastery", "monitor", "moped", "mortar", "mortarboard", "mosque", "mosquito_net", "motor_scooter", "mountain_bike", "mountain_tent", "mouse", "mousetrap", "moving_van", "muzzle", "nail", "neck_brace", "necklace", "nipple", "notebook", "obelisk", "oboe", "ocarina", "odometer", "oil_filter", "organ", "oscilloscope", "overskirt", "oxcart", "oxygen_mask", "packet", "paddle", "paddlewheel", "padlock", "paintbrush", "pajama", "palace", "panpipe", "paper_towel", "parachute", "parallel_bars", "park_bench", "parking_meter", "passenger_car", "patio", "pay-phone", "pedestal", "pencil_box", "pencil_sharpener", "perfume", "Petri_dish", "photocopier", "pick", "pickelhaube", "picket_fence", "pickup", "pier", "piggy_bank", "pill_bottle", "pillow", "ping-pong_ball", "pinwheel", "pirate", "pitcher", "plane", "planetarium", "plastic_bag", "plate_rack", "plow", "plunger", "Polaroid_camera", "pole", "police_van", "poncho", "pool_table", "pop_bottle", "pot", "potter's_wheel", "power_drill", "prayer_rug", "printer", "prison", "projectile", "projector", "puck", "punching_bag", "purse", "quill", "quilt", "racer", "racket", "radiator", "radio", "radio_telescope", "rain_barrel", "recreational_vehicle", "reel", "reflex_camera", "refrigerator", "remote_control", "restaurant", "revolver", "rifle", "rocking_chair", "rotisserie", "rubber_eraser", "rugby_ball", "rule", "running_shoe", "safe", "safety_pin", "saltshaker", "sandal", "sarong", "sax", "scabbard", "scale", "school_bus", "schooner", "scoreboard", "screen", "screw", "screwdriver", "seat_belt", "sewing_machine", "shield", "shoe_shop", "shoji", "shopping_basket", "shopping_cart", "shovel", "shower_cap", "shower_curtain", "ski", "ski_mask", "sleeping_bag", "slide_rule", "sliding_door", "slot", "snorkel", "snowmobile", "snowplow", "soap_dispenser", "soccer_ball", "sock", "solar_dish", "sombrero", "soup_bowl", "space_bar", "space_heater", "space_shuttle", "spatula", "speedboat", "spider_web", "spindle", "sports_car", "spotlight", "stage", "steam_locomotive", "steel_arch_bridge", "steel_drum", "stethoscope", "stole", "stone_wall", "stopwatch", "stove", "strainer", "streetcar", "stretcher", "studio_couch", "stupa", "submarine", "suit", "sundial", "sunglass", "sunglasses", "sunscreen", "suspension_bridge", "swab", "sweatshirt", "swimming_trunks", "swing", "switch", "syringe", "table_lamp", "tank", "tape_player", "teapot", "teddy", "television", "tennis_ball", "thatch", "theater_curtain", "thimble", "thresher", "throne", "tile_roof", "toaster", "tobacco_shop", "toilet_seat", "torch", "totem_pole", "tow_truck", "toyshop", "tractor", "trailer_truck", "tray", "trench_coat", "tricycle", "trimaran", "tripod", "triumphal_arch", "trolleybus", "trombone", "tub", "turnstile", "typewriter_keyboard", "umbrella", "unicycle", "upright", "vacuum", "vase", "vault", "velvet", "vending_machine", "vestment", "viaduct", "violin", "volleyball", "waffle_iron", "wall_clock", "wallet", "wardrobe", "warplane", "washbasin", "washer", "water_bottle", "water_jug", "water_tower", "whiskey_jug", "whistle", "wig", "window_screen", "window_shade", "Windsor_tie", "wine_bottle", "wing", "wok", "wooden_spoon", "wool", "worm_fence", "wreck", "yawl", "yurt", "web_site", "comic_book", "crossword_puzzle", "street_sign", "traffic_light", "book_jacket", "menu", "plate", "guacamole", "consomme", "hot_pot", "trifle", "ice_cream", "ice_lolly", "French_loaf", "bagel", "pretzel", "cheeseburger", "hotdog", "mashed_potato", "head_cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti_squash", "acorn_squash", "butternut_squash", "cucumber", "artichoke", "bell_pepper", "cardoon", "mushroom", "Granny_Smith", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "custard_apple", "pomegranate", "hay", "carbonara", "chocolate_sauce", "dough", "meat_loaf", "pizza", "potpie", "burrito", "red_wine", "espresso", "cup", "eggnog", "alp", "bubble", "cliff", "coral_reef", "geyser", "lakeside", "promontory", "sandbar", "seashore", "valley", "volcano", "ballplayer", "groom", "scuba_diver", "rapeseed", "daisy", "yellow_lady's_slipper", "corn", "acorn", "hip", "buckeye", "coral_fungus", "agaric", "gyromitra", "stinkhorn", "earthstar", "hen-of-the-woods", "bolete", "ear", "toilet_tissue"]

    pred_clean = pred_clean.map(lambda x: class_names[int(x)])
    pred_corrupt = pred_corrupt.map(lambda x: class_names[int(x)])
    
    def get_data(idx):
        if idx < threshold:
            return x_clean[idx]
        return x_corrupt[idx - threshold]

    def get_pred(idx):
        if idx < threshold:
            return pred_clean[idx]
        return pred_corrupt[idx - threshold]
    
    pairs_per_row = 4
    n_rows = len(positives) // pairs_per_row + (1 if len(positives) % pairs_per_row else 0)
    
    # Create figure with just the needed columns (2 per pair)
    fig, axes = plt.subplots(n_rows, pairs_per_row * 2, 
                            figsize=(2.5*pairs_per_row, 2*n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    for i, (clean_idx, corrupt_idx) in enumerate(positives):
        clean_img = get_data(clean_idx).permute(1, 2, 0).numpy()
        corrupt_img = get_data(corrupt_idx).permute(1, 2, 0).numpy()
        
        row = i // pairs_per_row
        col = (i % pairs_per_row) * 2  # Changed from *3 to *2
        
        axes[row, col].imshow(clean_img)
        axes[row, col].set_title(f'Image {clean_idx}: {get_pred(clean_idx)}', fontsize=6, pad=1)
        axes[row, col].axis('off')
        
        axes[row, col + 1].imshow(corrupt_img)
        axes[row, col + 1].set_title(f'Image {corrupt_idx}: {get_pred(corrupt_idx)}', fontsize=6, pad=1)
        axes[row, col + 1].axis('off')

    # Hide empty subplots
    for i in range(len(positives), pairs_per_row * n_rows):
        row = i // pairs_per_row
        col = (i % pairs_per_row) * 2  # Changed from *3 to *2
        axes[row, col].axis('off')
        axes[row, col + 1].axis('off')

    fig.suptitle(f"Flagged Inputs for semantic robustness on {dataset_corrupt}\n"
                 f"Corruption: {corruption} severity {severity}\n"
                 f"Model: {model}. {np} TP ({perc*100:.2f}%)", 
                 fontsize=10, y=1.0)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model}-{corruption}-{severity}.png', dpi=300, bbox_inches='tight')
    #plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize positive examples from a JSON file.')
    parser.add_argument('json_path', type=str, help='Path to the JSON file containing positive examples')
    parser.add_argument('--max_samples', type=int, default=12, help='Maximum number of samples to visualize')
    
    args = parser.parse_args()
    
    visualize_positives_from_json(args.json_path, max_samples=args.max_samples)


if __name__ == '__main__':
    main()
