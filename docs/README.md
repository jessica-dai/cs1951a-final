# data science blog final

## Final Blog Post (5/4/19)

### Vision

From our knowledge about public school funding, we thought there might be some correlation between demographics of a school -- in particular, the percentages of low-income  and students of color -- and the types and quality of extracurricular opportunities it provides. However, we found it difficult to find the correlation between demographics and school resources because of the lack of joinable attributes among the four datasets. Thus, we decided instead to predict the geographic region and community type based on various attributes from each of the datasets. While we didn’t explicitly follow our initial plans for this project, we still gleaned interesting information in this general topic area -- that is, that there do exist some relationships between school location within the US (northeast, southeast, central, west) and school community type (urban, suburban, town, rural), and the types of opportunities that are offered -- and furthermore, that these relationships are conditional on both location and community type. 

### DATA

We chose to examine four different data sets from data.gov, which included one on Arts education, one on Demographics and school resources (CRDC), one on Education Technology resources, and one on Career and Technical Education programs. The data varied in sizes; the largest dataset was the Demographics and school resources dataset, with over thousands of samples. The Education Technology and Arts datasets were much smaller. 
We were able to find distinct correlations between the datasets themselves, but we were not able to find correlations across datasets (explained more below). For preprocessing, we each individually went through our datasets and only saved the data that was relevant to our study. For example, the CRDC dataset included samples of private schools, boarding schools, and elementary schools, but we only wanted the data regarding public high schools. 
We struggled in finding a way to integrate our data; we considered joining them on what we thought were common attributes within the datasets, but we were not confident that the attributes were similar enough.  We ended up deciding to do something different, which is then explained more below.


### Methodology
At the beginning of the project, we hoped to be able to predict the demographics of a school given knowledge about its extracurricular programs, or vice versa. This required having complete information for each datapoint (i.e. each school), meaning that we would have had to been able to claim that the information contained in the arts dataset for a given school corresponded directly to information we could identify in the CTE program dataset for the same school. After working more with the individual datasets, we realized this would be impossible, given that all of our datasets were from separate surveys, conducted at different times, and hence did not survey the exact same schools and in fact had anonymized all entries. 

We then tried to find any general patterns or relationships at all within each dataset by running simple SQL queries to our database (for example, counting the number of entries with particular attributes) and checking whether those numbers suggested any correlation. This also proved difficult, especially because we wanted to be able to integrate insights from all datasets at a time. 

The factors and attributes that were common among all datasets was school region (northeast, southeast, central, and west) and community type (urban, suburban, town, and rural). The only dataset that didn’t have both was the CRDC dataset, which still collected information about school location (state), from which we could infer region. 

Given this, we decided to see whether we could predict region and community type based on information from a single dataset; then, to be able to combine information from all datasets, we would combine the best models for each dataset to make a final prediction. This therefore elides the issue of having to create “fake” or untrue data by joining entries across datasets. 

### Model design
For each dataset, there are a total of 16 possible classifications -- every possible combination of location and community type. The only exception is the CRDC dataset, with 4 possible classifications (just location). The classifications we used were as followed: 

1 -- city & northeast
2 -- city & southeast 
3 -- city & central
4 -- city & west
5 -- surburban & northeast
6 -- suburban & southeast
7 -- suburban & central
8 -- suburban & west
9 -- town & northeast
10 -- town & southeast
11 -- town & central
12 -- town & west
13 -- rural & northeast
14 -- rural & southeast
15 -- rural & central
16 -- rural & west 

Then, for each dataset, we train 5 of scikit-learn’s pre-implemented models: logistic regression, naive bayes (using the Bernoulli model), a neural network (using the multi-layer perceptron classifier), a SVM (using the linear SVC), and a decision tree. We use scikit-learn’s built-in train test split method, with default parameters of 75% training and 25% testing data.

The final predictive model is a vote of these five models, weighted by their accuracy.  The final prediction presumes we have information about the school’s demographics, arts, CTE, and edtech programs; then, we feed the relevant pieces of data to each separate model, 

The reason we chose to predict location and community type together, rather than separately, is that interestingly enough, the models perform better when predicting both at the same time than one by one. For instance, with the CTE dataset, predicting one at a time resulted in models that performed an average of 1.2x as good as guessing; predicting both together, however,  resulted in models that performed an average of 1.75x as good as guessing. These results validate our choice of using 16 classifications rather than 4, when possible. 

We recognize a glaring issue is the inability to test the final predictor for how well it does (for the very reason we chose to shift our strategy in the first place); that being said, this is the best we could do given our data, and in particular we actually end up performing much better than we thought we would (see below). 

Visualization, both for the preliminary exploration phase and for the modelling and result-generating phase, was done primarily via matplotlib, mostly because it was easiest and flexible to do so while working with numpy arrays in python. Additionally, d3 was used to make graphs that could be made interactive. 

### Results 

After training and testing our models, we found better-than-guessing correlations for each dataset. The probability of finding the correct classification for each dataset just by guessing would be 1nwhere n is the number of classifications. For the Arts, Technology, and Career datasets, we had 16 different classifications: 4 classifications describing location (Northeast, Southeast, West, Central)  and 4 classifications describing how developed the environment was. For the CRDC dataset, there were 4 classifications describing location, as there was no information given about how developed the school’s environment was. 

Our model accuracies are not the greatest, with the best accuracy being around 20%. However, the models are still meaningfully better than random guessing. The Arts dataset consistently yielded an accuracy around 3x better than random guessing, and each dataset had at least one model perform at least twice as well as random guessing. This could possible be the case because there were many attributes among the Arts dataset and the best combination of features was chosen to train the model. 


## blog post 1 -- 3/15/19

_What are the potential correlations between the demographics of public schools (income, race, dis/ability, ESL/ELL), and the resources/ opportunities available at those schools (edtech products, vocational/tech training programs, arts)?_

### Project introduction & overview

The most difficult component of this project so far has been deciding what kind of “stories” we might be able to tell with the data, based on not just what’s available, but also the common features between them; deciding on the “story” from even a preliminary scan of the data has been the general theme of our work for the past few weeks. 

We began by finding four of the most relevant datasets from government records, and splitting them up among our group members: school demographics (Amy), access to educational technology (Rebecca), career and technical education programs (Jessica), and arts education programs (YJ). Because these are all separate datasets, we thought that it would make sense to split them up “by person,” which would make the initial steps of understanding the basic information held within each dataset much easier. Furthermore, because all the datasets are set up/ formatted differently, this means that each of us should handle cleaning and preprocessing on our own datasets.

We’re combining information from a number of records/ studies/ surveys from the federal government, some of which are conducted by different agencies, which means that the schema are all formatted differently, and information types are often inconsistent (e.g. one survey will collect data on a per-school basis, and another will collect on a per-district basis). An additional difficulty in coming up with a coherent question or story is that many of these datasets are quite old and/or from different years, so patterns we find may not necessarily be accurate or relevant to the point in time that we’re trying to think about. That being said, an awareness of American history, geography, and resource distribution suggests that it’s unlikely that the content of these data changed drastically -- given that we don’t have additional resources available, we’re going to assume that this is the case but do so with the knowledge that we have incomplete information. 

The next two sections of the blog post will explain how we’re attempting to answer the question: what kinds of stories can we tell? What stories do we want to tell?

### Joinable Attributes

Given how disparate our datasets were, the first step was to find and decide on “joinable attributes” to link our datasets together. 

For each dataset, the most “relevant” attributes are listed below: 

* Demographics: state/location, school, enrollment
* Edtech: state/location, school region, race, community type, school enrollment size 
* CTE: region, community type (urban vs rural), district enrollment size
* Arts: school region (NE, SE, C, W), school community type (city, suburban, town, rural etc), school enrollment size, percent combined enrollment of black, hispanic, asian/pacific islander, or american indian/alaska native students in the school 

All of our datasets are national; region and enrollment are common among all datasets, but the detail and unit of region/enrollment are different. For instance, the dataset on Edtech divided the school enrollment size to these 3 categories: 1) less than 300 2) 300-999 3) 1000 or more, while the dataset on arts divided it to 1) less than 500 2) 500-999 3) 1000 or more. While we are able to easily combine the schools with 1000 or more students from both datasets to one, with the other two categories, things become more difficult. In this case, we can carefully make the assumption that schools with less than 500 students can encompass schools with less than 300 students and schools with 500-999 students is within the larger range of 300-999 students. We understand that making these generalizations can cause overlap in our data, especially when we set up our independent variables for further analysis (when building our linear regression model). We are open to all of your suggestions and will continue to explore new ways to combine and generalize our datasets.

Given this, our chosen “joins, ” ordered by priority, will be:
1. Location/region
2. Community type (rural/town/suburban/urban)
3. Enrollment

As noted above, enrollment is an important but especially finicky attribute. The example above indicates the difficulties in making approximations about enrollment types, but furthermore the units of evaluation are not constant across datasets (e.g. school vs district). To deal with this, one strategy we can apply is taking a percentile of both to provide an approximation for relative size. That being said, we’re not sure the extent to which enrollment would even be an explanatory variable -- absolute size of the district or school doesn’t necessarily indicate anything about the volume or distribution of resources within the school. If our results are too noisy after doing this, we may choose to remove enrollment. 

The rest of our analysis is predicated on the assumption that our chosen “joins” -- even when they don’t necessarily map to the same exact units or data points -- preserve most of the interesting information that we’re trying to extract, and entries from all four datasets where each of these attributes match will indicate information about very similar schools. We think that this is the case because though the data is represented differently, these are all nationwide surveys polling the same schools (or a random sample of those schools). 

That being said, we need to be careful about how exactly we do--or don’t--draw conclusions or correlations between these so-called “joinable attributes.” For instance, the arts education dataset and the CTE dataset have very joinable attributes, with two caveats: one, the arts dataset is school-wise, and the CTE dataset is districtwise; and two, the arts dataset also has a flag for the proportion of students of color. While we can make an approximation for which buckets to place each school or district into in terms of region and size, it doesn’t necessarily also make sense to draw a 1-1 mapping between matching region/size buckets and racial/ demographic information. 

Finally, we will pull in additional/ outside datasets that can link some attributes together. The US Census Bureau, for instance, has compiled the [Small Area Income and Poverty Estimates](https://www.census.gov/programs-surveys/saipe.html), which was used by the authors of the CTE survey (but frustratingly not included as a survey question!). Some preliminary calls to their API indicate that this dataset does contain some interesting information, in particular poverty levels for specific counties and school districts, which will not only allow us to create a baseline-level join for school-based data (which lists counties) and district-based data, but also provide an extra layer of explanatory information. 

### Features for analysis

Once we find a way to group the schools together, we will determine the best (most relevant features) among the datasets that best fit each classification method . For instance if the demographics or dis/ability of the classroom had a stronger correlation between certain metrics such as number of computers used, we would classify based on that feature.  Other classification methods such as linear regression can be used to determine whether a correlation exists between the relevant features and the dependent variables such as total time spent using computers. We will use the following classification models: 
Decision tree, linear regression, logistic regression, bayesian classification, SGD. 

The attributes that we’ll be looking at are:
* Enrollment demographics, particularly with breakdowns by AP enrollment and law enforcement referrals
* Computer usage by time & frequency
* Per-student computer access
* CTE program availability, and/or reasons for unavailability (the CTE dataset in particular has questions about whether financial need of schools or students play a role in the extent to which CTE programs are offered)
* Arts programs availability and participation rates

### Data cleaning, storage & representation

As noted above, each of us are cleaning our own datasets (since they have different shapes, number of attributes, etc.). That being said, we’re doing so in a standardized way, in particular, running python scripts to remove data with too many null values and then to organize the data. 

Our first, rough setup for data storage and representation is to dump everything into a SQL database, which will be useful and interesting for a high-level summary and overview of the relationship between different attributes we ant to look at. Each dataset will be a separate table, but we’re adding new columns in each table for processing the relevant joins as discussed above. 

As indicated in our pre-proposal, we hope our final product to have some sort of predictive component that encapsulates the relationships between different features. This means that we’ll want to create numeric schemata to translate qualitative attributes into something we can feed into statistical or numerical models. 

### Progress evaluation & next steps

By our next check-in after Spring Break, we will have finished constructing the SQL database, and created multiple visualizations of various complexities based on different queries to the database; furthermore, we should have sketched out some skeletons for what analytical models we’d like to use.

### Images

![DecisionTree](https://github.com/jessica-dai/cs1951a-final/tree/master/docs/decision_tree.png)
Description: Graph showing accuracies of each of the datasets trained on different models.

Image: avg_cp.png
Description: This graph demonstrates that region 13(Rural and Northeast) seems to have a significantly higher ratio of computers for instruction to total computers of around 0.6, which is a strange finding.  All the other 15 classes have ratios of around 0.4. Region 8(Suburban and West) has the lowest ratio.  

Image: avg_tech.png
Description: There seems to be no correlation between the 16 classes and the average amount of training. 

Image: gd_enr.png
Description: There seems to be a lower enrollment for districts in the central area and there seems to be greater variance in enrollment size for districts in the West. 

Image: program_qual.png
Description: Suburban schools seem to have greater variance in program quality score, as shown by the larger amount of schools two standard deviations above the mean. Furthermore, schools in the city had the highest probability of schools above the mean and rural schools had the least. 

Image: perc_cp.png
Description: This pie-chart was created with d3 and shows the percentage of computers for each region out of the total number of computers. Because each of the regions might have varying enrollment sizes, district size was used to normalize the ratios. Before normalization, there was greater variance in ratios in the data. The West region has the highest percent of computers in relation to district size and the NorthEast has the least. 

Image: train_vs_int.png
Description: The darker points represent greater density. The graph was plotted with a sample size of 1500 of the total data points to demonstrate greater clarity. Furthermore, the number of computers  was scaled down by 1/10. The data is clustered towards higher training and integration. There is greater variance among the number of computers, which makes sense because district enrollment sizes also vary. There is highest density in the upper right region, which seems to suggest that higher number of computers implies greater training and integration. 
