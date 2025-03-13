import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import pickle
import warnings
import tqdm
warnings.filterwarnings('ignore')

class IPLMatchPredictor:
    def __init__(self):
        self.batsman_stats = None
        self.bowler_stats = None
        self.fantasy_points_batsman = None
        self.fantasy_points_bowler = None
        self.match_data = None
        self.pressure_performance = None
        self.delivery_data = None
        
        self.team_encoder = LabelEncoder()
        self.player_encoder = LabelEncoder()
        self.venue_encoder = LabelEncoder()
        
        self.match_winner_model = None
        self.best_batsman_model = None
        self.best_bowler_model = None
        
        self.player_data = {}
        self.venue_data = {}
        self.team_data = {}
        self.player_vs_team = {}
        
    def load_data(self):
        print("Loading datasets...")
        
        # Load all required datasets
        self.batsman_stats = pd.read_csv('batsman_stats.csv')
        self.bowler_stats = pd.read_csv('bowler_stats.csv')
        self.fantasy_points_batsman = pd.read_csv('fantasy_points_batsman.csv')
        self.fantasy_points_bowler = pd.read_csv('fantasy_points_bowler.csv')
        self.match_data = pd.read_csv('new_match.csv')
        self.pressure_performance = pd.read_csv('overall_pressure_performance.csv')
        self.delivery_data = pd.read_csv('new_delivery.csv')
        
        print("Datasets loaded successfully!")
        
    def process_data(self):
        print("Processing data...")
        
        # Encode categorical variables
        all_teams = self.match_data['team1'].unique().tolist() + self.match_data['team2'].unique().tolist()
        self.team_encoder.fit(sorted(list(set(all_teams))))
        
        all_players = list(set(self.batsman_stats['batter'].tolist() + self.bowler_stats['bowler'].tolist()))
        self.player_encoder.fit(sorted(all_players))
        
        all_venues = sorted(list(self.match_data['city'].unique()))
        self.venue_encoder.fit(all_venues)
        
        # Create player statistics summary
        self.create_player_statistics()
        
        # Create venue statistics
        self.create_venue_statistics()
        
        # Create team statistics
        self.create_team_statistics()
        
        # Create player vs team statistics
        self.create_player_vs_team_statistics()
        
        print("Data processing completed!")
        
    def create_player_statistics(self):
        print("Creating player statistics...")
        
        # Process batsman data
        for _, row in tqdm.tqdm(self.batsman_stats.iterrows(), total=len(self.batsman_stats)):
            player_name = row['batter']
            
            if player_name not in self.player_data:
                self.player_data[player_name] = {
                    'batting': {}, 'bowling': {}, 'fantasy_points': {}
                }
            
            # Add batting statistics
            self.player_data[player_name]['batting'] = {
                'total_runs': row['total_runs'],
                'balls_faced': row['balls_faced'],
                'fours': row['fours'],
                'sixes': row['sixes'],
                'dismissals': row['dismissals'],
                'strike_rate': row['strike_rate'],
                'batting_average': row['batting_average'],
                'powerplay_runs': row['powerplay_runs'],
                'death_overs_runs': row['death_overs_runs']
            }
            
        # Process bowler data
        for _, row in tqdm.tqdm(self.bowler_stats.iterrows(), total=len(self.bowler_stats)):
            player_name = row['bowler']
            
            if player_name not in self.player_data:
                self.player_data[player_name] = {
                    'batting': {}, 'bowling': {}, 'fantasy_points': {}
                }
            
            # Add bowling statistics
            self.player_data[player_name]['bowling'] = {
                'total_wickets': row['total_wickets'],
                'total_runs_conceded': row['total_runs_conceded'],
                'balls_bowled': row['balls_bowled'],
                'economy_rate': row['economy_rate'],
                'bowling_average': row['bowling_average'],
                'strike_rate': row['strike_rate'],
                'powerplay_wickets': row['powerplay_wickets'],
                'death_overs_wickets': row['death_overs_wickets']
            }
            
        # Add fantasy points
        for _, row in tqdm.tqdm(self.fantasy_points_batsman.iterrows(), total=len(self.fantasy_points_batsman)):
            player_name = row['batter']
            if player_name in self.player_data:
                self.player_data[player_name]['fantasy_points']['batting'] = row['fantasy_points']
                
        for _, row in tqdm.tqdm(self.fantasy_points_bowler.iterrows(), total=len(self.fantasy_points_bowler)):
            player_name = row['bowler']
            if player_name in self.player_data:
                self.player_data[player_name]['fantasy_points']['bowling'] = row['fantasy_points']
        
        # Add pressure performance
        for _, row in tqdm.tqdm(self.pressure_performance.iterrows(), total=len(self.pressure_performance)):
            player_name = row['batter']
            if player_name in self.player_data:
                self.player_data[player_name]['pressure'] = {
                    'batting_pressure_rating': row['batting_pressure_rating'],
                    'bowling_pressure_rating': row['bowling_pressure_rating'],
                    'overall_pressure_rating': row['overall_pressure_rating']
                }
                
    def create_venue_statistics(self):
        print("Creating venue statistics...")
        
        # Group matches by venue
        venues = self.match_data['city'].unique()
        
        for venue in tqdm.tqdm(venues):
            venue_matches = self.match_data[self.match_data['city'] == venue]
            total_matches = len(venue_matches)
            
            # Calculate average scores at this venue
            venue_match_ids = venue_matches['id'].tolist()
            venue_deliveries = self.delivery_data[self.delivery_data['match_id'].isin(venue_match_ids)]
            
            # Calculate statistics by innings
            inning_stats = {}
            for inning in [1, 2]:
                inning_deliveries = venue_deliveries[venue_deliveries['inning'] == inning]
                if len(inning_deliveries) > 0:
                    total_runs = inning_deliveries['total_runs'].sum()
                    total_matches_with_inning = len(inning_deliveries['match_id'].unique())
                    avg_runs_per_match = total_runs / total_matches_with_inning if total_matches_with_inning > 0 else 0
                    
                    # Calculate wickets
                    wickets = inning_deliveries[inning_deliveries['is_wicket'] == 1].shape[0]
                    avg_wickets_per_match = wickets / total_matches_with_inning if total_matches_with_inning > 0 else 0
                    
                    inning_stats[inning] = {
                        'avg_runs': avg_runs_per_match,
                        'avg_wickets': avg_wickets_per_match
                    }
            
            # Calculate toss advantage
            toss_wins = venue_matches[venue_matches['toss_winner'] == venue_matches['winner']].shape[0]
            toss_win_percentage = (toss_wins / total_matches) * 100 if total_matches > 0 else 0
            
            # Calculate batting first advantage
            batting_first_wins = venue_matches[venue_matches['toss_decision'] == 'bat'].shape[0]
            batting_first_win_percentage = (batting_first_wins / total_matches) * 100 if total_matches > 0 else 0
            
            # Store venue data
            self.venue_data[venue] = {
                'total_matches': total_matches,
                'inning_stats': inning_stats,
                'toss_win_percentage': toss_win_percentage,
                'batting_first_win_percentage': batting_first_win_percentage
            }
            
    def create_team_statistics(self):
        print("Creating team statistics...")
        
        teams = sorted(list(set(self.match_data['team1'].unique().tolist() + self.match_data['team2'].unique().tolist())))
        
        for team in tqdm.tqdm(teams):
            # Matches where this team played
            team_matches_1 = self.match_data[self.match_data['team1'] == team]
            team_matches_2 = self.match_data[self.match_data['team2'] == team]
            team_matches = pd.concat([team_matches_1, team_matches_2])
            
            # Calculate win percentage
            team_wins = team_matches[team_matches['winner'] == team].shape[0]
            total_matches = len(team_matches)
            win_percentage = (team_wins / total_matches) * 100 if total_matches > 0 else 0
            
            # Calculate toss advantage
            toss_wins = team_matches[team_matches['toss_winner'] == team].shape[0]
            toss_win_percentage = (toss_wins / total_matches) * 100 if total_matches > 0 else 0
            
            # Calculate batting first vs chasing stats
            batting_first_matches = team_matches[(team_matches['toss_winner'] == team) & (team_matches['toss_decision'] == 'bat')]
            batting_first_wins = batting_first_matches[batting_first_matches['winner'] == team].shape[0]
            batting_first_win_percentage = (batting_first_wins / len(batting_first_matches)) * 100 if len(batting_first_matches) > 0 else 0
            
            chasing_matches = team_matches[(team_matches['toss_winner'] == team) & (team_matches['toss_decision'] == 'field')]
            chasing_wins = chasing_matches[chasing_matches['winner'] == team].shape[0]
            chasing_win_percentage = (chasing_wins / len(chasing_matches)) * 100 if len(chasing_matches) > 0 else 0
            
            # Store team data
            self.team_data[team] = {
                'total_matches': total_matches,
                'win_percentage': win_percentage,
                'toss_win_percentage': toss_win_percentage,
                'batting_first_win_percentage': batting_first_win_percentage,
                'chasing_win_percentage': chasing_win_percentage
            }
            
    def create_player_vs_team_statistics(self):
        print("Creating player vs team statistics...")
        
        teams = sorted(list(set(self.match_data['team1'].unique().tolist() + self.match_data['team2'].unique().tolist())))
        
        # Create player vs team stats from ball-by-ball data
        for _, delivery in tqdm.tqdm(self.delivery_data.iterrows(), total=len(self.delivery_data)):
            batter = delivery['batter']
            bowler = delivery['bowler']
            batting_team = delivery['batting_team']
            bowling_team = delivery['bowling_team']
            
            # Batter vs bowling team
            if batter not in self.player_vs_team:
                self.player_vs_team[batter] = {}
            
            if bowling_team not in self.player_vs_team[batter]:
                self.player_vs_team[batter][bowling_team] = {
                    'runs': 0, 'balls': 0, 'dismissals': 0, 
                    'fours': 0, 'sixes': 0, 'strike_rate': 0
                }
                
            self.player_vs_team[batter][bowling_team]['runs'] += delivery['batsman_runs']
            self.player_vs_team[batter][bowling_team]['balls'] += 1
            
            if delivery['is_wicket'] == 1 and delivery['player_dismissed'] == batter:
                self.player_vs_team[batter][bowling_team]['dismissals'] += 1
                
            if delivery['batsman_runs'] == 4:
                self.player_vs_team[batter][bowling_team]['fours'] += 1
                
            if delivery['batsman_runs'] == 6:
                self.player_vs_team[batter][bowling_team]['sixes'] += 1
                
            # Calculate strike rate
            if self.player_vs_team[batter][bowling_team]['balls'] > 0:
                self.player_vs_team[batter][bowling_team]['strike_rate'] = (
                    self.player_vs_team[batter][bowling_team]['runs'] / 
                    self.player_vs_team[batter][bowling_team]['balls']
                ) * 100
                
            # Bowler vs batting team
            if bowler not in self.player_vs_team:
                self.player_vs_team[bowler] = {}
                
            if batting_team not in self.player_vs_team[bowler]:
                self.player_vs_team[bowler][batting_team] = {
                    'wickets': 0, 'runs_conceded': 0, 'balls_bowled': 0,
                    'economy': 0, 'strike_rate': 0
                }
                
            self.player_vs_team[bowler][batting_team]['balls_bowled'] += 1
            self.player_vs_team[bowler][batting_team]['runs_conceded'] += (
                delivery['batsman_runs'] + delivery['extra_runs']
            )
            
            if delivery['is_wicket'] == 1:
                self.player_vs_team[bowler][batting_team]['wickets'] += 1
                
            # Calculate economy and strike rate
            if self.player_vs_team[bowler][batting_team]['balls_bowled'] >= 6:
                self.player_vs_team[bowler][batting_team]['economy'] = (
                    self.player_vs_team[bowler][batting_team]['runs_conceded'] / 
                    (self.player_vs_team[bowler][batting_team]['balls_bowled'] / 6)
                )
                
            if self.player_vs_team[bowler][batting_team]['wickets'] > 0:
                self.player_vs_team[bowler][batting_team]['strike_rate'] = (
                    self.player_vs_team[bowler][batting_team]['balls_bowled'] / 
                    self.player_vs_team[bowler][batting_team]['wickets']
                )
    
    def build_training_dataset(self):
        print("Building training dataset...")
        
        match_features = []
        match_outcomes = []
        best_batsman_features = []
        best_batsman_outcomes = []
        best_bowler_features = []
        best_bowler_outcomes = []
        
        # Loop through each match
        for _, match in tqdm.tqdm(self.match_data.iterrows(), total=len(self.match_data)):
            match_id = match['id']
            team1 = match['team1']
            team2 = match['team2']
            venue = match['city']
            winner = match['winner']
            toss_winner = match['toss_winner']
            toss_decision = match['toss_decision']
            
            # Skip matches with no clear winner
            if pd.isna(winner):
                continue
                
            # Get match deliveries
            match_deliveries = self.delivery_data[self.delivery_data['match_id'] == match_id]
            
            if len(match_deliveries) == 0:
                continue
                
            # Extract team players
            team1_batsmen = match_deliveries[match_deliveries['batting_team'] == team1]['batter'].unique().tolist()
            team1_bowlers = match_deliveries[match_deliveries['bowling_team'] == team1]['bowler'].unique().tolist()
            team2_batsmen = match_deliveries[match_deliveries['batting_team'] == team2]['batter'].unique().tolist()
            team2_bowlers = match_deliveries[match_deliveries['bowling_team'] == team2]['bowler'].unique().tolist()
            
            # Determine the best batsman in the match
            batsmen_runs = {}
            for _, delivery in match_deliveries.iterrows():
                batter = delivery['batter']
                runs = delivery['batsman_runs']
                
                if batter not in batsmen_runs:
                    batsmen_runs[batter] = 0
                    
                batsmen_runs[batter] += runs
                
            best_batsman = max(batsmen_runs.items(), key=lambda x: x[1])[0] if batsmen_runs else None
            
            # Determine the best bowler in the match
            bowler_wickets = {}
            bowler_runs_conceded = {}
            for _, delivery in match_deliveries.iterrows():
                bowler = delivery['bowler']
                is_wicket = delivery['is_wicket']
                runs_conceded = delivery['batsman_runs'] + delivery['extra_runs']
                
                if bowler not in bowler_wickets:
                    bowler_wickets[bowler] = 0
                    bowler_runs_conceded[bowler] = 0
                    
                if is_wicket == 1:
                    bowler_wickets[bowler] += 1
                    
                bowler_runs_conceded[bowler] += runs_conceded
                
            # Calculate bowling performance score (wickets * 10 - runs conceded)
            bowler_performance = {
                bowler: (bowler_wickets.get(bowler, 0) * 10) - bowler_runs_conceded.get(bowler, 0)
                for bowler in set(bowler_wickets.keys()) | set(bowler_runs_conceded.keys())
            }
            
            best_bowler = max(bowler_performance.items(), key=lambda x: x[1])[0] if bowler_performance else None
            
            # Skip if we don't have a best batsman or bowler
            if not best_batsman or not best_bowler:
                continue
                
            # Create feature set for match outcome
            team1_features = self.extract_team_features(team1)
            team2_features = self.extract_team_features(team2)
            venue_features = self.extract_venue_features(venue)
            
            # Create head-to-head features
            head_to_head_features = self.extract_head_to_head_features(team1, team2)
            
            # Combine all features for match outcome prediction
            match_feature = np.concatenate([
                team1_features, team2_features, venue_features, head_to_head_features,
                [1 if toss_winner == team1 else 0],  # Toss winner
                [1 if toss_decision == 'bat' else 0]  # Toss decision
            ])
            
            match_features.append(match_feature)
            match_outcomes.append(1 if winner == team1 else 0)  # 1 if team1 wins, 0 otherwise
            
            # Create feature set for best batsman prediction
            all_batsmen = set(team1_batsmen + team2_batsmen)
            for batter in all_batsmen:
                if batter not in self.player_data:
                    continue
                    
                player_features = self.extract_player_features(batter, is_batting=True)
                opposition_team = team2 if batter in team1_batsmen else team1
                vs_team_features = self.extract_player_vs_team_features(batter, opposition_team, is_batting=True)
                
                batsman_feature = np.concatenate([player_features, vs_team_features])
                best_batsman_features.append(batsman_feature)
                best_batsman_outcomes.append(1 if batter == best_batsman else 0)
                
            # Create feature set for best bowler prediction
            all_bowlers = set(team1_bowlers + team2_bowlers)
            for bowler in all_bowlers:
                if bowler not in self.player_data:
                    continue
                    
                player_features = self.extract_player_features(bowler, is_batting=False)
                opposition_team = team2 if bowler in team1_bowlers else team1
                vs_team_features = self.extract_player_vs_team_features(bowler, opposition_team, is_batting=False)
                
                bowler_feature = np.concatenate([player_features, vs_team_features])
                best_bowler_features.append(bowler_feature)
                best_bowler_outcomes.append(1 if bowler == best_bowler else 0)
                
        return (
            np.array(match_features), np.array(match_outcomes),
            np.array(best_batsman_features), np.array(best_batsman_outcomes),
            np.array(best_bowler_features), np.array(best_bowler_outcomes)
        )
        
    def extract_team_features(self, team):
        if team in self.team_data:
            team_info = self.team_data[team]
            return np.array([
                team_info['total_matches'],
                team_info['win_percentage'],
                team_info['toss_win_percentage'],
                team_info['batting_first_win_percentage'],
                team_info['chasing_win_percentage']
            ])
        else:
            return np.zeros(5)  # Default values if team not found
            
    def extract_venue_features(self, venue):
        if venue in self.venue_data:
            venue_info = self.venue_data[venue]
            inning_stats = venue_info['inning_stats']
            
            # Get average runs for each innings
            avg_runs_inning1 = inning_stats.get(1, {}).get('avg_runs', 0)
            avg_runs_inning2 = inning_stats.get(2, {}).get('avg_runs', 0)
            
            # Get average wickets for each innings
            avg_wickets_inning1 = inning_stats.get(1, {}).get('avg_wickets', 0)
            avg_wickets_inning2 = inning_stats.get(2, {}).get('avg_wickets', 0)
            
            return np.array([
                venue_info['total_matches'],
                venue_info['toss_win_percentage'],
                venue_info['batting_first_win_percentage'],
                avg_runs_inning1,
                avg_runs_inning2,
                avg_wickets_inning1,
                avg_wickets_inning2
            ])
        else:
            return np.zeros(7)  # Default values if venue not found
            
    def extract_head_to_head_features(self, team1, team2):
        head_to_head_matches = self.match_data[
            ((self.match_data['team1'] == team1) & (self.match_data['team2'] == team2)) |
            ((self.match_data['team1'] == team2) & (self.match_data['team2'] == team1))
        ]
        
        total_matches = len(head_to_head_matches)
        if total_matches > 0:
            team1_wins = head_to_head_matches[head_to_head_matches['winner'] == team1].shape[0]
            team1_win_percentage = (team1_wins / total_matches) * 100
            
            batting_first_wins = head_to_head_matches[
                ((head_to_head_matches['team1'] == team1) & (head_to_head_matches['winner'] == team1) & (head_to_head_matches['toss_decision'] == 'bat')) |
                ((head_to_head_matches['team2'] == team1) & (head_to_head_matches['winner'] == team1) & (head_to_head_matches['toss_decision'] == 'field'))
            ].shape[0]
            
            batting_first_win_percentage = (batting_first_wins / team1_wins) * 100 if team1_wins > 0 else 0
            
            return np.array([total_matches, team1_win_percentage, batting_first_win_percentage])
        else:
            return np.zeros(3)  # Default values if no head-to-head matches
            
    def extract_player_features(self, player, is_batting=True):
        if player in self.player_data:
            player_info = self.player_data[player]
            
            if is_batting:
                batting_stats = player_info.get('batting', {})
                pressure_stats = player_info.get('pressure', {})
                
                return np.array([
                    batting_stats.get('total_runs', 0),
                    batting_stats.get('balls_faced', 0),
                    batting_stats.get('fours', 0),
                    batting_stats.get('sixes', 0),
                    batting_stats.get('strike_rate', 0),
                    batting_stats.get('batting_average', 0),
                    batting_stats.get('powerplay_runs', 0),
                    batting_stats.get('death_overs_runs', 0),
                    pressure_stats.get('batting_pressure_rating', 0),
                    player_info.get('fantasy_points', {}).get('batting', 0)
                ])
            else:
                bowling_stats = player_info.get('bowling', {})
                pressure_stats = player_info.get('pressure', {})
                
                return np.array([
                    bowling_stats.get('total_wickets', 0),
                    bowling_stats.get('total_runs_conceded', 0),
                    bowling_stats.get('balls_bowled', 0),
                    bowling_stats.get('economy_rate', 0),
                    bowling_stats.get('bowling_average', 0),
                    bowling_stats.get('strike_rate', 0),
                    bowling_stats.get('powerplay_wickets', 0),
                    bowling_stats.get('death_overs_wickets', 0),
                    pressure_stats.get('bowling_pressure_rating', 0),
                    player_info.get('fantasy_points', {}).get('bowling', 0)
                ])
        else:
            return np.zeros(10)  # Default values if player not found
            
    def extract_player_vs_team_features(self, player, team, is_batting=True):
        if player in self.player_vs_team and team in self.player_vs_team[player]:
            player_vs_team_info = self.player_vs_team[player][team]
            
            if is_batting:
                return np.array([
                    player_vs_team_info.get('runs', 0),
                    player_vs_team_info.get('balls', 0),
                    player_vs_team_info.get('dismissals', 0),
                    player_vs_team_info.get('fours', 0),
                    player_vs_team_info.get('sixes', 0),
                    player_vs_team_info.get('strike_rate', 0)
                ])
            else:
                return np.array([
                    player_vs_team_info.get('wickets', 0),
                    player_vs_team_info.get('runs_conceded', 0),
                    player_vs_team_info.get('balls_bowled', 0),
                    player_vs_team_info.get('economy', 0),
                    player_vs_team_info.get('strike_rate', 0)
                ])
        else:
            return np.zeros(6 if is_batting else 5)  # Default values if player_vs_team not found
            
    def train_models(self):
        print("Training models...")
        
        # Build training datasets
        match_X, match_y, batsman_X, batsman_y, bowler_X, bowler_y = self.build_training_dataset()
        
        # Split data into training and testing sets
        match_X_train, match_X_test, match_y_train, match_y_test = train_test_split(
            match_X, match_y, test_size=0.2, random_state=42
        )
        
        batsman_X_train, batsman_X_test, batsman_y_train, batsman_y_test = train_test_split(
            batsman_X, batsman_y, test_size=0.2, random_state=42
        )
        
        bowler_X_train, bowler_