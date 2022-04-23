# 6.6 Pretrained Convolutional Network And Image Processing

# 1. Code

## 1.1 VGG16

Using VGG16 network.

```python
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

from PIL import Image


def visualOriginal(im):
    imarray = np.asarray(im) / 255.0
    plt.figure()
    plt.imshow(imarray)
    plt.show()

def visualMiddleFeatures(layer,num_features):
    plt.figure(figsize=(11,6))
    for ii in range(num_features):
        # visualize every image
        plt.subplot(6,11,ii+1)
        plt.imshow(layer.data.numpy()[0,ii,:,:],
                   cmap='gray')
        plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def visualDeeperLayer(layer,num_features):
    plt.figure(figsize=(12,6))
    for ii in range(num_features):
        # visualize every image
        plt.subplot(6,12,ii+1)
        plt.imshow(layer.data.numpy()[0,ii,:,:],
                   cmap='gray')
        plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


vgg16 = models.vgg16(pretrained=True)
im =Image.open('data/Image/Collections/elephant.jpg')

data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

input_im = data_transforms(im).unsqueeze(0)
print('input_im.shape: ',input_im.shape)

activation = {} # save output from different layers
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# get the feature from middle convolutional layer
vgg16.eval()
# get features after first max pooling in the fourth layer
vgg16.features[4].register_forward_hook(get_activation('maxpool1'))
_ = vgg16(input_im)
maxpool1 = activation['maxpool1']
print('Size of features after max pooling : ', maxpool1.shape)
# visualize features
visualMiddleFeatures(maxpool1,maxpool1.shape[1])

# get features from deeper convolutional layers
vgg16.eval()
vgg16.features[21].register_forward_hook(get_activation('conv_21'))
_ = vgg16(input_im)
conv_21 = activation['conv_21']
print('Size of features: ', conv_21.shape)
# visualize top 72 features
# visualMiddleFeatures(conv_21,72)
# visualDeeperLayer(conv_21,72)


# get 1000 labels when VGG16 is training
#LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/'
#  response = requests.get(LABELS_URL)

# print(labels)
# labels = {int(key): value for key,value in response.json().items()}

LABEL_PATH = 'data/labels_ImageNet.json'
with open(LABEL_PATH) as j:
    labels_orig = eval(j.read())
print(labels_orig)
labels = {}
for ii in range(len(labels_orig)):
    labels.update({ii:labels_orig[ii]})

# print(labels)
# prediction using VGG16
vgg16.eval()
im_pre = vgg16(input_im)
softmax = nn.Softmax(dim=1)
im_pre_prob = softmax(im_pre)
prob,prelab = torch.topk(im_pre_prob, 5)
prob = prob.data.numpy().flatten()
prelab = prelab.numpy().flatten()
for ii, lab in enumerate(prelab):
    print('index: ',lab,' label: ',labels[lab],' || ',prob[ii])

```
## 1.2 Custom Network

Using our own network.

```python
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

from PIL import Image


def visualCombine(heatmap):
    img = cv2.imread('data/Image/Collections/elephant.jpg')
    heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    Grad_cam_img = heatmap * 0.4 + img
    Grad_cam_img = Grad_cam_img / Grad_cam_img.max()

    # visualize the image
    b,g,r = cv2.split(Grad_cam_img)
    Grad_cam_img = cv2.merge([r,g,b])
    plt.figure()
    plt.imshow(Grad_cam_img)
    plt.show()

class MyVgg16(nn.Module):
    def __init__(self):
        super(MyVgg16, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.features_conv = self.vgg.features[:30]
        self.max_pool = self.vgg.features[30]
        self.avgpool = self.vgg.avgpool
        self.classifier = self.vgg.classifier
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        h = x.register_hook(self.activations_hook)
        x = self.max_pool(x)
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)


vgg16 = models.vgg16(pretrained=True)
im =Image.open('data/Image/Collections/elephant.jpg')

data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

input_im = data_transforms(im).unsqueeze(0)
print('input_im.shape: ',input_im.shape)

LABEL_PATH = 'data/labels_ImageNet.json'
with open(LABEL_PATH) as j:
    labels_orig = eval(j.read())
print(labels_orig)
labels = {}
for ii in range(len(labels_orig)):
    labels.update({ii:labels_orig[ii]})

activation = {} # save output from different layers
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

vggcam = MyVgg16()
vggcam.eval()
im_pre = vggcam(input_im)

softmax = nn.Softmax(dim=1)
im_pre_prob = softmax(im_pre)
prob,prelab = torch.topk(im_pre_prob, 5)
prob = prob.data.numpy().flatten()
prelab = prelab.numpy().flatten()
for ii, lab in enumerate(prelab):
    print('index: ',lab,' label: ',labels[lab],' || ',prob[ii])


im_pre[:,prelab[0]].backward()
gradients = vggcam.get_activations_gradient()
mean_gradients = torch.mean(gradients,dim=[0,2,3])
activations = vggcam.get_activations(input_im).detach()
for i in range(len(mean_gradients)):
    activations[:,i,:,:] *= mean_gradients[i]

heatmap = torch.mean(activations,dim=1).squeeze()
heatmap = F.relu(heatmap)
heatmap /= torch.max(heatmap)
heatmap = heatmap.numpy()

# plt.matshow(heatmap)
# plt.show()

# visualCombine(heatmap)
```

# 2. Illustration

Here we use the following image to illustrate.

![image](Images/6_22_Elephant(Original).png)


input_im.shape:  torch.Size([1, 3, 224, 224])

Size of features after max pooling :  torch.Size([1, 64, 112, 112])

![image](Images/6_23_Features-of-the-fourth-layer.png)

Size of features of the deeper layer:  torch.Size([1, 512, 28, 28])

![image](Images/6_24_Features-of-the-deeper-layer.png)

Labels:
```python
['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark', 'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco', 'indigo bunting', 'American robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'American dipper', 'kite', 'bald eagle', 'vulture', 'great grey owl', 'fire salamander', 'smooth newt', 'newt', 'spotted salamander', 'axolotl', 'American bullfrog', 'tree frog', 'tailed frog', 'loggerhead sea turtle', 'leatherback sea turtle', 'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'green iguana', 'Carolina anole', 'desert grassland whiptail lizard', 'agama', 'frilled-necked lizard', 'alligator lizard', 'Gila monster', 'European green lizard', 'chameleon', 'Komodo dragon', 'Nile crocodile', 'American alligator', 'triceratops', 'worm snake', 'ring-necked snake', 'eastern hog-nosed snake', 'smooth green snake', 'kingsnake', 'garter snake', 'water snake', 'vine snake', 'night snake', 'boa constrictor', 'African rock python', 'Indian cobra', 'green mamba', 'sea snake', 'Saharan horned viper', 'eastern diamondback rattlesnake', 'sidewinder', 'trilobite', 'harvestman', 'scorpion', 'yellow garden spider', 'barn spider', 'European garden spider', 'southern black widow', 'tarantula', 'wolf spider', 'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie grouse', 'peacock', 'quail', 'partridge', 'grey parrot', 'macaw', 'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'duck', 'red-breasted merganser', 'goose', 'black swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala', 'wombat', 'jellyfish', 'sea anemone', 'brain coral', 'flatworm', 'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton', 'chambered nautilus', 'Dungeness crab', 'rock crab', 'fiddler crab', 'red king crab', 'American lobster', 'spiny lobster', 'crayfish', 'hermit crab', 'isopod', 'white stork', 'black stork', 'spoonbill', 'flamingo', 'little blue heron', 'great egret', 'bittern', 'crane (bird)', 'limpkin', 'common gallinule', 'American coot', 'bustard', 'ruddy turnstone', 'dunlin', 'common redshank', 'dowitcher', 'oystercatcher', 'pelican', 'king penguin', 'albatross', 'grey whale', 'killer whale', 'dugong', 'sea lion', 'Chihuahua', 'Japanese Chin', 'Maltese', 'Pekingese', 'Shih Tzu', 'King Charles Spaniel', 'Papillon', 'toy terrier', 'Rhodesian Ridgeback', 'Afghan Hound', 'Basset Hound', 'Beagle', 'Bloodhound', 'Bluetick Coonhound', 'Black and Tan Coonhound', 'Treeing Walker Coonhound', 'English foxhound', 'Redbone Coonhound', 'borzoi', 'Irish Wolfhound', 'Italian Greyhound', 'Whippet', 'Ibizan Hound', 'Norwegian Elkhound', 'Otterhound', 'Saluki', 'Scottish Deerhound', 'Weimaraner', 'Staffordshire Bull Terrier', 'American Staffordshire Terrier', 'Bedlington Terrier', 'Border Terrier', 'Kerry Blue Terrier', 'Irish Terrier', 'Norfolk Terrier', 'Norwich Terrier', 'Yorkshire Terrier', 'Wire Fox Terrier', 'Lakeland Terrier', 'Sealyham Terrier', 'Airedale Terrier', 'Cairn Terrier', 'Australian Terrier', 'Dandie Dinmont Terrier', 'Boston Terrier', 'Miniature Schnauzer', 'Giant Schnauzer', 'Standard Schnauzer', 'Scottish Terrier', 'Tibetan Terrier', 'Australian Silky Terrier', 'Soft-coated Wheaten Terrier', 'West Highland White Terrier', 'Lhasa Apso', 'Flat-Coated Retriever', 'Curly-coated Retriever', 'Golden Retriever', 'Labrador Retriever', 'Chesapeake Bay Retriever', 'German Shorthaired Pointer', 'Vizsla', 'English Setter', 'Irish Setter', 'Gordon Setter', 'Brittany', 'Clumber Spaniel', 'English Springer Spaniel', 'Welsh Springer Spaniel', 'Cocker Spaniels', 'Sussex Spaniel', 'Irish Water Spaniel', 'Kuvasz', 'Schipperke', 'Groenendael', 'Malinois', 'Briard', 'Australian Kelpie', 'Komondor', 'Old English Sheepdog', 'Shetland Sheepdog', 'collie', 'Border Collie', 'Bouvier des Flandres', 'Rottweiler', 'German Shepherd Dog', 'Dobermann', 'Miniature Pinscher', 'Greater Swiss Mountain Dog', 'Bernese Mountain Dog', 'Appenzeller Sennenhund', 'Entlebucher Sennenhund', 'Boxer', 'Bullmastiff', 'Tibetan Mastiff', 'French Bulldog', 'Great Dane', 'St. Bernard', 'husky', 'Alaskan Malamute', 'Siberian Husky', 'Dalmatian', 'Affenpinscher', 'Basenji', 'pug', 'Leonberger', 'Newfoundland', 'Pyrenean Mountain Dog', 'Samoyed', 'Pomeranian', 'Chow Chow', 'Keeshond', 'Griffon Bruxellois', 'Pembroke Welsh Corgi', 'Cardigan Welsh Corgi', 'Toy Poodle', 'Miniature Poodle', 'Standard Poodle', 'Mexican hairless dog', 'grey wolf', 'Alaskan tundra wolf', 'red wolf', 'coyote', 'dingo', 'dhole', 'African wild dog', 'hyena', 'red fox', 'kit fox', 'Arctic fox', 'grey fox', 'tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat', 'Egyptian Mau', 'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar', 'lion', 'tiger', 'cheetah', 'brown bear', 'American black bear', 'polar bear', 'sloth bear', 'mongoose', 'meerkat', 'tiger beetle', 'ladybug', 'ground beetle', 'longhorn beetle', 'leaf beetle', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant', 'grasshopper', 'cricket', 'stick insect', 'cockroach', 'mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly', 'damselfly', 'red admiral', 'ringlet', 'monarch butterfly', 'small white', 'sulphur butterfly', 'gossamer-winged butterfly', 'starfish', 'sea urchin', 'sea cucumber', 'cottontail rabbit', 'hare', 'Angora rabbit', 'hamster', 'porcupine', 'fox squirrel', 'marmot', 'beaver', 'guinea pig', 'common sorrel', 'zebra', 'pig', 'wild boar', 'warthog', 'hippopotamus', 'ox', 'water buffalo', 'bison', 'ram', 'bighorn sheep', 'Alpine ibex', 'hartebeest', 'impala', 'gazelle', 'dromedary', 'llama', 'weasel', 'mink', 'European polecat', 'black-footed ferret', 'otter', 'skunk', 'badger', 'armadillo', 'three-toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'siamang', 'guenon', 'patas monkey', 'baboon', 'macaque', 'langur', 'black-and-white colobus', 'proboscis monkey', 'marmoset', 'white-headed capuchin', 'howler monkey', 'titi', "Geoffroy's spider monkey", 'common squirrel monkey', 'ring-tailed lemur', 'indri', 'Asian elephant', 'African bush elephant', 'red panda', 'giant panda', 'snoek', 'eel', 'coho salmon', 'rock beauty', 'clownfish', 'sturgeon', 'garfish', 'lionfish', 'pufferfish', 'abacus', 'abaya', 'academic gown', 'accordion', 'acoustic guitar', 'aircraft carrier', 'airliner', 'airship', 'altar', 'ambulance', 'amphibious vehicle', 'analog clock', 'apiary', 'apron', 'waste container', 'assault rifle', 'backpack', 'bakery', 'balance beam', 'balloon', 'ballpoint pen', 'Band-Aid', 'banjo', 'baluster', 'barbell', 'barber chair', 'barbershop', 'barn', 'barometer', 'barrel', 'wheelbarrow', 'baseball', 'basketball', 'bassinet', 'bassoon', 'swimming cap', 'bath towel', 'bathtub', 'station wagon', 'lighthouse', 'beaker', 'military cap', 'beer bottle', 'beer glass', 'bell-cot', 'bib', 'tandem bicycle', 'bikini', 'ring binder', 'binoculars', 'birdhouse', 'boathouse', 'bobsleigh', 'bolo tie', 'poke bonnet', 'bookcase', 'bookstore', 'bottle cap', 'bow', 'bow tie', 'brass', 'bra', 'breakwater', 'breastplate', 'broom', 'bucket', 'buckle', 'bulletproof vest', 'high-speed train', 'butcher shop', 'taxicab', 'cauldron', 'candle', 'cannon', 'canoe', 'can opener', 'cardigan', 'car mirror', 'carousel', 'tool kit', 'carton', 'car wheel', 'automated teller machine', 'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello', 'mobile phone', 'chain', 'chain-link fence', 'chain mail', 'chainsaw', 'chest', 'chiffonier', 'chime', 'china cabinet', 'Christmas stocking', 'church', 'movie theater', 'cleaver', 'cliff dwelling', 'cloak', 'clogs', 'cocktail shaker', 'coffee mug', 'coffeemaker', 'coil', 'combination lock', 'computer keyboard', 'confectionery store', 'container ship', 'convertible', 'corkscrew', 'cornet', 'cowboy boot', 'cowboy hat', 'cradle', 'crane (machine)', 'crash helmet', 'crate', 'infant bed', 'Crock Pot', 'croquet ball', 'crutch', 'cuirass', 'dam', 'desk', 'desktop computer', 'rotary dial telephone', 'diaper', 'digital clock', 'digital watch', 'dining table', 'dishcloth', 'dishwasher', 'disc brake', 'dock', 'dog sled', 'dome', 'doormat', 'drilling rig', 'drum', 'drumstick', 'dumbbell', 'Dutch oven', 'electric fan', 'electric guitar', 'electric locomotive', 'entertainment center', 'envelope', 'espresso machine', 'face powder', 'feather boa', 'filing cabinet', 'fireboat', 'fire engine', 'fire screen sheet', 'flagpole', 'flute', 'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster bed', 'freight car', 'French horn', 'frying pan', 'fur coat', 'garbage truck', 'gas mask', 'gas pump', 'goblet', 'go-kart', 'golf ball', 'golf cart', 'gondola', 'gong', 'gown', 'grand piano', 'greenhouse', 'grille', 'grocery store', 'guillotine', 'barrette', 'hair spray', 'half-track', 'hammer', 'hamper', 'hair dryer', 'hand-held computer', 'handkerchief', 'hard disk drive', 'harmonica', 'harp', 'harvester', 'hatchet', 'holster', 'home theater', 'honeycomb', 'hook', 'hoop skirt', 'horizontal bar', 'horse-drawn vehicle', 'hourglass', 'iPod', 'clothes iron', "jack-o'-lantern", 'jeans', 'jeep', 'T-shirt', 'jigsaw puzzle', 'pulled rickshaw', 'joystick', 'kimono', 'knee pad', 'knot', 'lab coat', 'ladle', 'lampshade', 'laptop computer', 'lawn mower', 'lens cap', 'paper knife', 'library', 'lifeboat', 'lighter', 'limousine', 'ocean liner', 'lipstick', 'slip-on shoe', 'lotion', 'speaker', 'loupe', 'sawmill', 'magnetic compass', 'mail bag', 'mailbox', 'tights', 'tank suit', 'manhole cover', 'maraca', 'marimba', 'mask', 'match', 'maypole', 'maze', 'measuring cup', 'medicine chest', 'megalith', 'microphone', 'microwave oven', 'military uniform', 'milk can', 'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl', 'mobile home', 'Model T', 'modem', 'monastery', 'monitor', 'moped', 'mortar', 'square academic cap', 'mosque', 'mosquito net', 'scooter', 'mountain bike', 'tent', 'computer mouse', 'mousetrap', 'moving van', 'muzzle', 'nail', 'neck brace', 'necklace', 'nipple', 'notebook computer', 'obelisk', 'oboe', 'ocarina', 'odometer', 'oil filter', 'organ', 'oscilloscope', 'overskirt', 'bullock cart', 'oxygen mask', 'packet', 'paddle', 'paddle wheel', 'padlock', 'paintbrush', 'pajamas', 'palace', 'pan flute', 'paper towel', 'parachute', 'parallel bars', 'park bench', 'parking meter', 'passenger car', 'patio', 'payphone', 'pedestal', 'pencil case', 'pencil sharpener', 'perfume', 'Petri dish', 'photocopier', 'plectrum', 'Pickelhaube', 'picket fence', 'pickup truck', 'pier', 'piggy bank', 'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate ship', 'pitcher', 'hand plane', 'planetarium', 'plastic bag', 'plate rack', 'plow', 'plunger', 'Polaroid camera', 'pole', 'police van', 'poncho', 'billiard table', 'soda bottle', 'pot', "potter's wheel", 'power drill', 'prayer rug', 'printer', 'prison', 'projectile', 'projector', 'hockey puck', 'punching bag', 'purse', 'quill', 'quilt', 'race car', 'racket', 'radiator', 'radio', 'radio telescope', 'rain barrel', 'recreational vehicle', 'reel', 'reflex camera', 'refrigerator', 'remote control', 'restaurant', 'revolver', 'rifle', 'rocking chair', 'rotisserie', 'eraser', 'rugby ball', 'ruler', 'running shoe', 'safe', 'safety pin', 'salt shaker', 'sandal', 'sarong', 'saxophone', 'scabbard', 'weighing scale', 'school bus', 'schooner', 'scoreboard', 'CRT screen', 'screw', 'screwdriver', 'seat belt', 'sewing machine', 'shield', 'shoe store', 'shoji', 'shopping basket', 'shopping cart', 'shovel', 'shower cap', 'shower curtain', 'ski', 'ski mask', 'sleeping bag', 'slide rule', 'sliding door', 'slot machine', 'snorkel', 'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball', 'sock', 'solar thermal collector', 'sombrero', 'soup bowl', 'space bar', 'space heater', 'space shuttle', 'spatula', 'motorboat', 'spider web', 'spindle', 'sports car', 'spotlight', 'stage', 'steam locomotive', 'through arch bridge', 'steel drum', 'stethoscope', 'scarf', 'stone wall', 'stopwatch', 'stove', 'strainer', 'tram', 'stretcher', 'couch', 'stupa', 'submarine', 'suit', 'sundial', 'sunglass', 'sunglasses', 'sunscreen', 'suspension bridge', 'mop', 'sweatshirt', 'swimsuit', 'swing', 'switch', 'syringe', 'table lamp', 'tank', 'tape player', 'teapot', 'teddy bear', 'television', 'tennis ball', 'thatched roof', 'front curtain', 'thimble', 'threshing machine', 'throne', 'tile roof', 'toaster', 'tobacco shop', 'toilet seat', 'torch', 'totem pole', 'tow truck', 'toy store', 'tractor', 'semi-trailer truck', 'tray', 'trench coat', 'tricycle', 'trimaran', 'tripod', 'triumphal arch', 'trolleybus', 'trombone', 'tub', 'turnstile', 'typewriter keyboard', 'umbrella', 'unicycle', 'upright piano', 'vacuum cleaner', 'vase', 'vault', 'velvet', 'vending machine', 'vestment', 'viaduct', 'violin', 'volleyball', 'waffle iron', 'wall clock', 'wallet', 'wardrobe', 'military aircraft', 'sink', 'washing machine', 'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle', 'wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle', 'wing', 'wok', 'wooden spoon', 'wool', 'split-rail fence', 'shipwreck', 'yawl', 'yurt', 'website', 'comic book', 'crossword', 'traffic sign', 'traffic light', 'dust jacket', 'menu', 'plate', 'guacamole', 'consomme', 'hot pot', 'trifle', 'ice cream', 'ice pop', 'baguette', 'bagel', 'pretzel', 'cheeseburger', 'hot dog', 'mashed potato', 'cabbage', 'broccoli', 'cauliflower', 'zucchini', 'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber', 'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'custard apple', 'pomegranate', 'hay', 'carbonara', 'chocolate syrup', 'dough', 'meatloaf', 'pizza', 'pot pie', 'burrito', 'red wine', 'espresso', 'cup', 'eggnog', 'alp', 'bubble', 'cliff', 'coral reef', 'geyser', 'lakeshore', 'promontory', 'shoal', 'seashore', 'valley', 'volcano', 'baseball player', 'bridegroom', 'scuba diver', 'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'rose hip', 'horse chestnut seed', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn mushroom', 'earth star', 'hen-of-the-woods', 'bolete', 'ear', 'toilet paper']
```

Using VGG16:

```python
index:  386  label:  African bush elephant  ||  0.8033338
index:  101  label:  tusker  ||  0.1595378
index:  385  label:  Asian elephant  ||  0.03709441
index:  354  label:  dromedary  ||  1.331438e-05
index:  291  label:  lion  ||  1.1394195e-05
```

Using our own network:

```python
index:  386  label:  African bush elephant  ||  0.8033338
index:  101  label:  tusker  ||  0.1595378
index:  385  label:  Asian elephant  ||  0.03709441
index:  354  label:  dromedary  ||  1.331438e-05
index:  291  label:  lion  ||  1.1394195e-05
```

Heatmap:

![image](Images/6_26_Heatmap.png)

Mix of heatmap and original image:

![image](Images/6_27_Mix-of-heatmap-and-original-image.png)
