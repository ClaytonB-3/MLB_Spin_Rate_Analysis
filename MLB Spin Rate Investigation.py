

from pybaseball import cache
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import ticker
import pybaseball
from pybaseball import statcast
import plotly.express as px
style.use(style='fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
pybaseball.cache.enable()
cache.enable()

# Collect data on every pitch thrown through July 12 using pybaseball library
df = statcast(start_dt='2021-04-01', end_dt='2021-07-12')

df.head()

print('There have been', len(df.index), 'pitches thrown as of July 9th.')

df.columns

relevent_col = ['pitch_type', 'game_date', 'release_speed',
                'release_pos_x', 'release_pos_z', 'player_name',
                'zone', 'des', 'p_throws', 'home_team', 'away_team',
                'plate_x', 'plate_z', 'inning_topbot', 'launch_speed',
                'launch_angle', 'effective_speed', 'release_spin_rate',
                'release_extension', 'estimated_ba_using_speedangle',
                'estimated_woba_using_speedangle', 'woba_value', 'woba_denom',
                'babip_value', 'iso_value', 'launch_speed_angle',
                'at_bat_number', 'pitch_number', 'pitch_name', 'spin_axis']
df = df[relevent_col]

df.info()

df.isnull().sum()

# Remove rows with nulls indicating a lack of a reading
df = df[~df['release_spin_rate'].isnull()]
df = df[~df['pitch_type'].isnull()]
df = df[~df['pitch_name'].isnull()]
df = df[~df['zone'].isnull()]

# Create Bauer Units (b units) column
df['b_units'] = (df['release_spin_rate']
                 / df['release_speed'])
# Create month and week columns
df['month'] = df['game_date'].dt.month
df['week'] = df['game_date'].dt.isocalendar().week

# Update week numbers with date of Monday of that week
week_date_mapper = {
                    13: 'March 29',
                    14: 'April 05',
                    15: 'April 12',
                    16: 'April 19',
                    17: 'April 26',
                    18: 'May 03',
                    19: 'May 10',
                    20: 'May 17',
                    21: 'May 24',
                    22: 'May 31',
                    23: 'June 07',
                    24: 'June 14',
                    25: 'June 21',
                    26: 'June 28',
                    27: 'July 05'
                    }
df['week'] = df['week'].replace(week_date_mapper)
df['week'] = df['week'] + ', 2021'
df['week of'] = df['week'].apply(lambda x: dt.strptime(x, '%B %d, %Y'))
df.drop('week', axis='columns', inplace=True)
df['week of'].head(10)

# Add column for team of the pitching team
# If pitching during bottom of inning then away team
df['pitch_team'] = df['home_team']
df.loc[df['inning_topbot'] == 'Bot', 'pitch_team'] = df.loc[
    df['inning_topbot'] == 'Bot', 'away_team']
df[['away_team', 'home_team', 'inning_topbot', 'pitch_team']].head()

# Investigate pitch frequency
pitch_count = (df.groupby('pitch_name')['release_spin_rate']
               .count().reset_index().sort_values(
                'release_spin_rate',
                ascending=False))
pitch_count.rename(
                   mapper={'release_spin_rate': 'count'},
                   axis='columns',
                   inplace=True
                   )
pitch_count['frequency'] = round((pitch_count['count']
                                 / pitch_count['count'].sum())*100, 2)
pitch_count


# We have five pitch names that occur less than 0.2% of the time. Two of these,
# fastball and 2-seam Fastball are likely from a legacy system or misnamed as
# there are a high volume of the common names for the three main types of
# fastballs: 4-Seam Fastball, Sinker, and Cutter. Knuckleball, Eephus, and
# Screwball occur at a very low rate in the modern game. I will remove all
# rows of these five types.

df = df[~df['pitch_name'].isin(
     ['Fastball', 'Knuckleball', 'Eephus', '2-Seam Fastball', 'Screwball']
     )]

pitch_count = (df.groupby('pitch_name')['release_spin_rate']
               .count().reset_index().sort_values('release_spin_rate',
                                                  ascending=False))
pitch_count.rename(mapper={'release_spin_rate': 'count'},
                   axis='columns', inplace=True)
pitch_count['frequency'] = round((pitch_count['count']
                                 / pitch_count['count'].sum()
                                  )*100, 2)
pitch_count


# If there was a drop in spinrate, when did it occur?
# Group by day, week, and month and average spin rate for each, then visualize.
# Create daily, weekly, and month average spin rate dataframes
daily_avg_spin = (df.groupby('game_date')
                  ['release_spin_rate'].mean().reset_index())
daily_avg_spin['game_date_str'] = (daily_avg_spin['game_date']
                                   .apply(lambda x: x.strftime('%Y-%m-%d')))
week_avg_spin = df.groupby('week of')['release_spin_rate'].mean().reset_index()
monthly_avg_spin = (df.groupby('month')
                    ['release_spin_rate'].mean().reset_index()
                    )
# Monthly spinrate
all_monthly_line = sns.lineplot(data=monthly_avg_spin,
                                x='month',
                                y='release_spin_rate')
plt.title('Average Spin Rate Per Month - All Pitches (2021)',
          fontsize=16, fontweight=600)
plt.ylabel('Average Spin Rate', fontsize=12, labelpad=10)
plt.xlabel('Month', fontsize=12, labelpad=10)

plt.show()


# Weekly spinrate
positions = [week for week in week_avg_spin['week of']]
labels = [week_date_mapper[key] for key in week_date_mapper]

all_weekly_line = sns.lineplot(data=week_avg_spin, x='week of',
                               y='release_spin_rate')
plt.title('Average Spin Rate Per Week - All Pitches (2021)',
          fontsize=16, fontweight=600)
plt.ylabel('Average Spin Rate', fontsize=12, labelpad=10)
plt.xlabel('Week of', fontsize=12, labelpad=10)
plt.xticks(rotation=70)
plt.show()


# Daily spinrate

# Adjust x axis labels
# Retrieve index in dataframe of first of each month for labeling x axis
positions = [daily_avg_spin['game_date_str'][daily_avg_spin['game_date_str']
                                             == '2021-04-01'].index[0],
             daily_avg_spin['game_date_str'][daily_avg_spin['game_date_str']
                                             == '2021-05-01'].index[0],
             daily_avg_spin['game_date_str'][daily_avg_spin['game_date_str']
                                             == '2021-06-01'].index[0],
             daily_avg_spin['game_date_str'][daily_avg_spin['game_date_str']
                                             == '2021-07-01'].index[0]]
labels = ['April 1', 'May 1', 'June 1', 'July 1']

fig, ax = plt.subplots(figsize=(20, 12))

all_daily_line = sns.lineplot(data=daily_avg_spin,
                              x='game_date_str',
                              y='release_spin_rate',
                              ax=ax)
plt.title('Average Spin Rate Per Day - All Pitches (2021)',
          fontsize=24, fontweight=600)
plt.ylabel('Average Spin Rate', fontsize=24, labelpad=10)
plt.xlabel('', fontsize=12, labelpad=10)
# Set x axis tick labels
all_daily_line.xaxis.set_major_locator(ticker.FixedLocator(positions))
all_daily_line.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
# x and y tick fontsize
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.show()

# Look for date in early June with a drop in spin

# Create dataframe of average spin rate per day June 1 - June 15
june_1_15_daily_spin = daily_avg_spin[
                       (daily_avg_spin['game_date'] >= '2021-05-31')
                       &
                       (daily_avg_spin['game_date'] <= '2021-06-15')
                                     ]
june_1_15_daily_spin


# Viz early June

june_daily_line = sns.lineplot(data=june_1_15_daily_spin,
                               x='game_date_str',
                               y='release_spin_rate')
plt.title('Average Spin Rate Per Day - All Pitches (Early June)',
          fontsize=16,
          fontweight=600)
plt.ylabel('Average Spin Rate', fontsize=12, labelpad=10)
plt.xlabel('', fontsize=12, labelpad=10)
plt.xticks(rotation=90)
plt.show()

# Did the frequency of pitches thrown change?

# Create df of frequency thrown for each pitch before June 7
pre_freq = (df[
           df['game_date'] <= '2021-06-7'
             ].groupby('pitch_name')['release_spin_rate'].count()
            .reset_index().sort_values('release_spin_rate', ascending=False))
pre_freq.rename(mapper={'release_spin_rate': 'pre_count'},
                axis='columns',
                inplace=True)
pre_freq['pre_frequency'] = round((pre_freq['pre_count']
                                  / pre_freq['pre_count'].sum()
                                   )*100, 2)
# Create df of frequency thrown for each pitch after June 7
post_freq = (df[df['game_date'] > '2021-06-7']
             .groupby('pitch_name')['release_spin_rate'].count().reset_index()
             .rename(mapper={'release_spin_rate': 'post_count'},
                     axis='columns'))
post_freq['post_frequency'] = round((post_freq['post_count']
                                    / post_freq['post_count'].sum()
                                     )*100, 2)
# Merge before and after dataframes and calculate difference
pre_post_freq = pre_freq.merge(post_freq,
                               how='left',
                               left_on='pitch_name', right_on='pitch_name')
pre_post_freq['freq_diff'] = (pre_post_freq['post_frequency']
                              - pre_post_freq['pre_frequency'])
pre_post_freq['freq_diff_pct_freq'] = round((pre_post_freq['freq_diff']
                                             / pre_post_freq['pre_frequency']
                                             ) * 100, 2)
pre_post_freq


# The largest drop in frequency of a pitch thrown as a percent of it's
# frequency thrown in the Knuckle Curve, followed by the cutter and changeup.
# Let's take a look at the spin rates of these different pitches.

# Create df of each pitch spin rate pre June 7
pre_spin = (df[df['game_date'] <= '2021-06-7']
            .groupby('pitch_name')['release_spin_rate'].mean()
            .reset_index().sort_values('release_spin_rate', ascending=False))
pre_spin.rename(mapper={'release_spin_rate': 'pre_spin'},
                axis='columns', inplace=True)
# Create df of each pitch spin rate post June 7
post_spin = (df[df['game_date'] > '2021-06-7']
             .groupby('pitch_name')['release_spin_rate'].mean()
             .reset_index().rename(mapper={'release_spin_rate': 'post_spin'},
                                   axis='columns'))
# Merge df and calc differences
pre_post_spin = pre_spin.merge(post_spin, how='left',
                               left_on='pitch_name',
                               right_on='pitch_name')
pre_post_spin['spin_diff'] = (pre_post_spin['post_spin']
                              - pre_post_spin['pre_spin'])
pre_post_spin['spin_diff_pct_spin'] = (pre_post_spin['spin_diff']
                                       / pre_post_spin['pre_spin']
                                       ) * 100

pre_post_spin


# Across the board we see between a 2.5 and 3.5% drop in spin rate, so it
# appears as though likely no matter what players are throwing they would be
# effected by this rule change if they are using sticky stuff. The split-finger
# saw the largest drop in spin rate as a percent of spin, but it is also the
# least thrown and slowest spinning pitching, so it is likely highly suceptable
# to questionable spin readings and a low sample size.

# ### Which Teams are Impacted the Most?
# Was this a problem equally spread throughout the league, or did it vary team
# to team? I'll calclate the difference in average spin per team before and
# after June 7.

# Calculate pre, post, and difference in spin for each team
team_pre_spin = (df[df['game_date'] <= '2021-06-07']
                 .groupby(by='pitch_team')['release_spin_rate'].mean()
                 .reset_index().rename(
                                      mapper={'release_spin_rate': 'pre_spin'},
                                      axis='columns')
                 )
team_post_spin = (df[df['game_date'] > '2021-06-07']
                  .groupby(by='pitch_team')['release_spin_rate'].mean()
                  .reset_index().rename(
                             mapper={'release_spin_rate': 'post_spin'},
                             axis='columns'))
team_spin_comp = team_pre_spin.merge(team_post_spin,
                                     how='left',
                                     left_on='pitch_team',
                                     right_on='pitch_team')
team_spin_comp['Difference'] = (team_spin_comp['post_spin']
                                - team_spin_comp['pre_spin'])
team_spin_comp = team_spin_comp.sort_values('Difference')
team_spin_comp

# Use plotly to visualize difference for each team
fig = px.bar(team_spin_comp, x='pitch_team', y='Difference',
             hover_name='pitch_team', hover_data=['pre_spin', 'post_spin'])

fig.show()


# Create df to compare spin pre and post June 7
pitcher_pre_spin = (df[df['game_date'] < '2021-06-07']
                    .groupby(by='player_name')['release_spin_rate'].mean()
                    .reset_index().rename(
                    mapper={'release_spin_rate': 'Pre_Spin'},
                    axis='columns'))
pitcher_post_spin = (df[df['game_date'] >= '2021-06-07']
                     .groupby(by='player_name')['release_spin_rate'].mean()
                     .reset_index().rename(
                     mapper={'release_spin_rate': 'Post_Spin'},
                     axis='columns'))
pitcher_spin_comp = pitcher_pre_spin.merge(pitcher_post_spin,
                                           how='left',
                                           left_on='player_name',
                                           right_on='player_name')
pitcher_spin_comp['spin_diff'] = (pitcher_spin_comp['Post_Spin']
                                  - pitcher_spin_comp['Pre_Spin'])
pitcher_spin_comp = pitcher_spin_comp.sort_values('spin_diff')
pitcher_spin_comp


# Total pitches thrown by each pitcher
pitcher_grouper = df.groupby('player_name')
pitches_thrown = pitcher_grouper['pitch_name'].count()
pitches_thrown = pitches_thrown.reset_index()
pitches_thrown = pitches_thrown.rename(mapper={
                'pitch_name': 'pitches_thrown'
                }, axis='columns')
pitches_thrown


# dfs of appearances by each pitcher before and since June 7
apps_after_grouper = (df[df['game_date'] >= '2021-06-07']
                      .groupby(by='player_name'))
apps_after = apps_after_grouper.agg({'game_date': 'nunique'}).reset_index()
apps_after = apps_after.rename(mapper={
                'game_date': 'appearances_june_7_later'
                }, axis='columns')
apps_before_grouper = (df[df['game_date'] < '2021-06-07']
                       .groupby(by='player_name'))
apps_before = apps_before_grouper.agg({'game_date': 'nunique'}).reset_index()
apps_before = apps_before.rename(mapper={
                'game_date': 'appearances_june_7_before'
                                        }, axis='columns')
# merge together with inner join to only get players with appearances
# before and after
apps = apps_after.merge(apps_before,
                        how='inner',
                        left_on='player_name',
                        right_on='player_name')
apps['total_apps'] = (apps['appearances_june_7_later']
                      + apps['appearances_june_7_before'])
apps


# Merge apps and pitches dataframes
pitches_apps = apps.merge(pitches_thrown,
                          how='left',
                          left_on='player_name',
                          right_on='player_name')
pitches_apps['pitches_per_app'] = round(pitches_apps['pitches_thrown']
                                        / pitches_apps['total_apps'], 1)
pitches_apps

# Merge apps and pitches dataframe with spin dataframe
apps_spin_diff = pitches_apps.merge(pitcher_spin_comp,
                                    how='left',
                                    left_on='player_name',
                                    right_on='player_name')
apps_spin_diff.head(5)


# The last two dataframes I will make is a comparison of xwOBA and Bauer units
# (spin rate/velocity)) before and after June 7th for each pitcher. xwOBA
# is up first.

# Create df to compare xwOBA pre and post June 7
pitcher_pre_xwoba = (df[
                     df['game_date'] < '2021-06-07']
                     .groupby('player_name')['estimated_woba_using_speedangle']
                     .mean().reset_index().rename(
                     mapper={'estimated_woba_using_speedangle': 'Pre_xwOBA'},
                     axis='columns'))
pitcher_post_xwoba = (df[
                      df['game_date'] >= '2021-06-07']
                      .groupby('player_name')
                      ['estimated_woba_using_speedangle'].mean().reset_index()
                      .rename(
                      mapper={'estimated_woba_using_speedangle': 'Post_xwOBA'},
                      axis='columns'))
pitcher_xwoba_comp = pitcher_pre_xwoba.merge(pitcher_post_xwoba,
                                             how='left',
                                             left_on='player_name',
                                             right_on='player_name')
pitcher_xwoba_comp['xwOBA_diff'] = (pitcher_xwoba_comp['Post_xwOBA']
                                    - pitcher_xwoba_comp['Pre_xwOBA'])
pitcher_xwoba_comp = pitcher_xwoba_comp.sort_values('xwOBA_diff')
pitcher_xwoba_comp.head()

# Merge xwOBA dataframe with with full comp dataframe
apps_spin_xwoba = apps_spin_diff.merge(pitcher_xwoba_comp,
                                       how='left',
                                       left_on='player_name',
                                       right_on='player_name')
apps_spin_xwoba.head(5)

# Create df to compare Bauer units pre and post June 7
pitcher_pre_bu = (df[df['game_date'] < '2021-06-07'].groupby(by='player_name')
                  ['b_units'].mean().reset_index()
                  .rename(mapper={'b_units': 'Pre_b_units'}, axis='columns'))
pitcher_post_bu = (df[df['game_date'] >= '2021-06-07'].groupby('player_name')
                   ['b_units'].mean().reset_index()
                   .rename(mapper={'b_units': 'Post_b_units'}, axis='columns'))
pitcher_bu_comp = pitcher_pre_bu.merge(pitcher_post_bu,
                                       how='left',
                                       left_on='player_name',
                                       right_on='player_name')
pitcher_bu_comp['bu_diff'] = (pitcher_bu_comp['Post_b_units']
                              - pitcher_bu_comp['Pre_b_units'])
pitcher_bu_comp = pitcher_bu_comp.sort_values('bu_diff')
pitcher_bu_comp.head()

# Merge b unit comparison to ful dataframe
apps_spin_xwoba_bu = apps_spin_xwoba.merge(pitcher_bu_comp,
                                           how='left',
                                           left_on='player_name',
                                           right_on='player_name')
apps_spin_xwoba_bu.head(5)

# Calculate correlations - All Pitchers
all_spin_xwoba_corr = apps_spin_xwoba_bu[
  ['xwOBA_diff', 'pitches_per_app', 'Pre_Spin', 'spin_diff', 'bu_diff']
  ].corr()

# Plot the correlations - All Pitchers
fig, ax = plt.subplots(figsize=(12, 10))
att_heatmap = sns.heatmap(all_spin_xwoba_corr)
plt.title("xwOBA Correlations - All Pitchers",
          fontsize=24, fontweight=750, pad=15)
plt.yticks(rotation=45, fontsize=14)
plt.xticks(rotation=45, fontsize=14)


plt.show()

# Create dataframe of just starting pitchers
healthy_starters = apps_spin_xwoba_bu[
    (apps_spin_xwoba_bu['appearances_june_7_later'] >= 5)
    &
    (apps_spin_xwoba_bu['appearances_june_7_before'] >= 5)
    &
    (apps_spin_xwoba_bu['pitches_per_app'] >= 75)
                                ]
print('There are',
      len(healthy_starters.index),
      'pitchers that meet our criteria.')
healthy_starters

# Calculate correlations - Starters
starters_spin_xwoba_corr = healthy_starters[
  ['xwOBA_diff', 'pitches_per_app', 'Pre_Spin', 'spin_diff', 'bu_diff']
  ].corr()

# Plot the correlations - Starters
fig, ax = plt.subplots(figsize=(12, 10))
att_heatmap = sns.heatmap(starters_spin_xwoba_corr)
plt.title("xwOBA Correlations - Starters", fontsize=24, fontweight=750, pad=15)
plt.yticks(rotation=45, fontsize=14)
plt.xticks(rotation=45, fontsize=14)


plt.show()


# Once again there aren't any strong correlations, so we'll take a look at
# relievers (pitchers who have appeared in at least 5 games before and 5 games
# after Juen 7 and average less than 50 pitches an outing).

# Create dataframe of relievers
healthy_relievers = apps_spin_xwoba_bu[
    (apps_spin_xwoba_bu['appearances_june_7_later'] >= 5)
    &
    (apps_spin_xwoba_bu['appearances_june_7_before'] >= 5)
    &
    (apps_spin_xwoba_bu['pitches_per_app'] <= 50)
                                ]
print('There are',
      len(healthy_relievers.index),
      'pitchers that meet our criteria.')
healthy_relievers.head()

# Calculate correlations - Relievers
relievers_spin_xwoba_corr = healthy_relievers[
 ['xwOBA_diff', 'pitches_per_app', 'Pre_Spin', 'spin_diff', 'bu_diff']
 ].corr()

# Plot the correlations - Relievers
fig, ax = plt.subplots(figsize=(12, 10))
att_heatmap = sns.heatmap(relievers_spin_xwoba_corr)
plt.title("xwOBA Correlations - Relievers",
          fontsize=24, fontweight=750, pad=15)
plt.yticks(rotation=45, fontsize=14)
plt.xticks(rotation=45, fontsize=14)


plt.show()


# Once again no strong correlations, so lastly I'll take a look to see if the
# top 50 pitchers by loss of b units who meet either our starter or reliever
# criteria have been impacted by their loss in spin.

# Pitchers who meet starter or reliever criteria
starters_relievers = apps_spin_xwoba_bu[
    ((apps_spin_xwoba_bu['appearances_june_7_later'] >= 5)
     &
     (apps_spin_xwoba_bu['appearances_june_7_before'] >= 5)
     &
     (apps_spin_xwoba_bu['pitches_per_app'] >= 75))
    |
    ((apps_spin_xwoba_bu['appearances_june_7_later'] >= 5)
     &
     (apps_spin_xwoba_bu['appearances_june_7_before'] >= 5)
     &
     (apps_spin_xwoba_bu['pitches_per_app'] <= 50))
                                ]
len(starters_relievers.index)


# Pitchers who lost the 50 most b units
top_50_bu_loss = starters_relievers.sort_values('bu_diff')[0:50]


# Calculate correlations - Top b unit loss
top_50_spin_xwoba_corr = top_50_bu_loss[
 ['xwOBA_diff', 'pitches_per_app', 'Pre_Spin', 'spin_diff', 'bu_diff']
 ].corr()

# Plot the correlations - Top b unit loss
fig, ax = plt.subplots(figsize=(12, 10))
att_heatmap = sns.heatmap(top_50_spin_xwoba_corr)
plt.title("xwOBA Correlations - Top BU Loss",
          fontsize=24, fontweight=750, pad=15)
plt.yticks(rotation=45, fontsize=14)
plt.xticks(rotation=45, fontsize=14)

plt.show()
