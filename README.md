# MLB_spin_rate_investigation

## Intro
### The "sticky" Background
Over the last several years spin rate has become a much talked about statistic amongst baseball analysts and fans. As spin rates rose, pitches got nastier and nastier, and strikeouts rose, the hardest thing to do in professional sports, hitting a baseball, got a little too hard. Curveballs, sliders, and other breaking pitches unsurprisingly got sharper, but fastballs also benefitted from the introduction of sticky stuff. An increased spin rate on a four-seam fastball caused fastballs to have the appearance of rising, in reality they were just falling less than a lifetime of knowledge told the hitter it should. All the while there was an understanding throughout the sport that one of the primary factors in this rise was the use of "sticky stuff" by pitchers to get an unnaturally strong grip on the ball. That may or may not seem like a huge issue to some, after all isn't having a good grip on something you're trying to throw 100 mph important? However, it is a direct violation of rules 3.01, 6.02(c), and 6.02(d) regarding the use of outside substances to influence the flight of the baseball. This rule was never enforced for a variety of reasons, but on June 05 2021 reports [emerged](https://www.espn.com/mlb/story/_/id/31572769/mlb-plans-enforcement-foreign-substance-rules-being-finalized-rollout-pending-sources-say) that the league started investigating the issue and warned players change was coming. On June 21, 2021 Major League Baseball officially started enforcing it by checking pitchers gloves, hats, and belts (where they would often store patches of sticky stuff for use) and ejecting them and suspending them if they find sticky stuff.  

### Questions/Goals
1. Did this change in rule enforcement cause a drop in spin rate and if so which pitch types were most impacted?
2. If there was a change in spin rate when did it occur? On the threat of enforcement, or the actual enforcement itself. 
3. Did some teams see a larger drop in spin rate than others?
4. Has there been a change in xwOBA amongst pitchers since spin rate has dropped.

## Preparing Python
1. Install Python 3 via Anaconda or other method. 
2. In command line navigate to the directory containing the requirements text file and run `pip install -r requirements_spin.txt` to install the required packages.

## Running the Code
1. Clone the repo for this project to your computer ([instructions])(https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository)
2. Run the python file `MLB Spin Rate Investigation.py` from the repo

## Viewing Existing Analysis
I've uploaded my Jupyter Notebook to the repo, so if you just want to look at the code with more detailed explainations take a look at that file (` MLB Spin Rate Investigation.ipynb`)

## Data Source
The data for this project was sourced from the statcast data made availbale on [baseballsavant.mlb.com](https://baseballsavant.mlb.com/statcast_search) via their search function (linked). The data was collected using the python library [pybaseball](https://github.com/jldbc/pybaseball)
