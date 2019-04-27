# Art Auction Valuation

## Motivation
Last November, Leonardo da Vinci's painting "Saviour of the World," sold at Christie's in New York for $450.3 million, shattering the world record for any work of art sold at auction. As an artist myself, I have been curious about how the commercial art auction market evaluates art pieces, so I turned my curiosity and passion for art into this data science project. The goal of the project is to classify if an artwork by 7 famous artists will be sold for more than $20,000, and if an artwork from less known artists will be sold for more than $2,000.

## Data Description
Web scraped art auction data. The dataset consists of 37,638 art pieces sold at a total valuation of $9.47 billion. Sold prices include a maximum of $119.92 million and a minimum of $3. Since the sold price gap between famous artists and small artists is very wide, I divided dataset into two subsets: 7 famous artists and 7399 less known artists. I built machine learning models and used different combinations of features for each subset. For this demonstration, I will primarily focus on the 7 famous artists model.

An overview of features and missing data through a heat map:
<img src='https://github.com/jasonshi10/art_auction_valuation/blob/master/images/miss_data.png'>
- 50% of sold time data is missing
- Some year made and artists death year data are missing.

## Features

| Feature Name  | Description  | Example  |
| ------------ | ------------ | ------------ |
| Artist  | Artist's name  | Pablo Picasso  |
|  Country |  Artist's country of origin |  Spain |
| Year of Birth  | Artist's brith year  | 1891  |
| Year of Death  | Artist's death year  | 1903  |
| Name | Name of the artwork  | 'Untitled'  |
| Material  | Material used for an artwork  |  Oil on canvas |
| Height  | Height of an artwork in inches | 24  |
| Width  | Width of an artwork in inches  | 12  |
| Link  | Link of the artworks' images  |   |
| Source  | Where the data is scraped from  |   |
| Dominant Color  |  Dominant color in an artwork | Green  |
| Brightness  | Mean brightness of an artwork. A value closer to 0 denotes a dark image and that closer to 255 indicates a bright one |  129 |
|Ratio of Unique Colors | The number of unique colors in an image as a ration of the total number of pixels  | 0.21  |
|Threshold Black Percentage   | If pixel value is greater than a threshold value(here we use 127, range from 0-255), it is assigned one value (255,white), else it is assigned another value (0,black). Then calculate the percentage of white or black in the image, and get the ratio of black pixels in the greyscale of paintings  | 99.87  |
|High Brightness Percentage   | Calculate the average brightness of each paintings and how many pixels have two times of the average brightness, then get ratio of these two numbers. | 0.00  |
|Low Brightness Percentage   |Calculate the average brightness of each paintings and count how many pixels have less than half of the mean brightness of that image, then get ratio of these two numbers.   | 0.00  |
|Corner Percentage   | Use Harris Corner Detection algorithm to detect the corner in the artworks. Corner is the intersection of two edges, it represents a point in which the directions of these two edges change. Hence, the gradient of the image (in both directions) have a high variation, which can be used to detect it. With that, we can calculate the ratio of pixels as corners in the full image.  | 0.31  |
|Edge Percentage   | Use Canny Edge Detection algorithm to detect the edges in the image. And then calculate the percentage of pixels recognized as edges in the whole picture.  | 12.08  |
|Face Count   |  Number of faces in an artwork's images | 3  |
|Sold Time   | When the auction sales happened  | 2007-05-03  |

## Feature Extraction
Use OpenCV and Python Image Library (PIL) to quantify aesthetics and extract features like dominant color, mean brightness, face count, etc.

## Model Used
- Logistic regression
- Random Forest
- Gradient Boosting
- Adaptive Boosting
- Voting Classifier

## Model Comparison and Results
<img src='https://github.com/jasonshi10/art_auction_valuation/blob/master/images/models.png' width="620" height="340">
Ensemble of three tree models performs the best with 80% accuracy.

## Key Features
Partial dependence from Gradient Boosting indicates key features that contribute to the prediction:
<img src='https://github.com/jasonshi10/art_auction_valuation/blob/master/images/height.png' width="300" height="250">
<img src='https://github.com/jasonshi10/art_auction_valuation/blob/master/images/unique.png' width="300" height="250">
<img src="https://github.com/jasonshi10/art_auction_valuation/blob/master/images/corner.png" width="300" height="250">

- The height of an artwork and unique color ratio are positively correlated with the price sold.
- Corner percentage is negatively correlated with the price sold.

## Additional Notes
I was wondering why my model only achieved 80% of accuracy and did some detective work. There are some other hard to quantify features that heavily influence the value of an artwork such as:
- General sentiment towards certain art styles change over time, which may influence auction prices.
- Artists who had successful exhibitions prior to an auction tend to sell for higher prices.
- Collectors may buy artworks to avoid taxes, providing an incentive to pay more.
- During auctions, competition between bidders raises final selling prices.
- Art value is about more than aesthetics, intangible factors such as conceptual value often play a role in a work’s value. Also the rarity of a work.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Numpy
- Matplotlib
- Python Image Library(PIL)
- AWS EC2
- OpenCV
- Plotly

## Reference

Dataset is provided by github.com/ahmedhosny/theGreenCanvas
