#2nd
# Part 1: Imports and Initial Setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import warnings
from scipy import stats
from tqdm import tqdm
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

class TurbofanAnalysis:
    def __init__(self):
        self.datasets = {}
        self.models = {}
        self.scalers = {}
        self.sensor_cols = None
        self.setting_cols = None

    def load_data(self, dataset_id):
        """
        Load the Turbofan Engine dataset files
        """
        try:
            # Define column names
            cols = ['unit', 'cycle'] + [f'setting{i}' for i in range(1,4)]
            cols.extend([f'sensor{i}' for i in range(1,22)])

            # Load training data
            train_df = pd.read_csv(f'CMaps/train_{dataset_id}.txt',
                                 sep='\s+', header=None, names=cols)

            # Load test data and RUL data
            test_df = pd.read_csv(f'CMaps/test_{dataset_id}.txt',
                                sep='\s+', header=None, names=cols)
            rul_df = pd.read_csv(f'CMaps/RUL_{dataset_id}.txt',
                               sep='\s+', header=None, names=['RUL'])

            # Store data in the datasets dictionary
            self.datasets[dataset_id] = {
                'train': train_df,
                'test': test_df,
                'rul': rul_df
            }

            # Define sensor and setting columns if not already defined
            if self.sensor_cols is None:
                self.sensor_cols = [f'sensor{i}' for i in range(1,22)]
                self.setting_cols = [f'setting{i}' for i in range(1,4)]

            print(f"Training data shape: {train_df.shape}")
            print(f"Test data shape: {test_df.shape}")
            print(f"RUL data shape: {rul_df.shape}")

            return True

        except Exception as e:
            print(f"Error loading dataset {dataset_id}: {str(e)}")
            return False

    def dataset_analysis(self, dataset_id='FD001'):
        """
        Comprehensive analysis of a specific dataset
        """
        if dataset_id not in self.datasets:
            print(f"Dataset {dataset_id} not loaded")
            return False

        print(f"\nAnalyzing Dataset {dataset_id}")
        train_df = self.datasets[dataset_id]['train']

        # Basic statistics
        print("\nBasic Statistics for Sensor Measurements:")
        stats_df = train_df[self.sensor_cols].describe()
        print(stats_df)

        # Operating conditions analysis
        print("\nOperating Conditions Analysis:")
        operating_conditions = train_df[self.setting_cols].drop_duplicates()
        print(f"Number of unique operating conditions: {len(operating_conditions)}")
        print("\nUnique operating conditions:")
        print(operating_conditions)

        # Unit/Engine analysis
        n_units = train_df['unit'].nunique()
        avg_cycles = train_df.groupby('unit')['cycle'].max().mean()
        print(f"\nNumber of engines: {n_units}")
        print(f"Average cycles per engine: {avg_cycles:.2f}")

        # Visualizations
        self._plot_sensor_distributions(train_df, dataset_id)
        self._plot_sensor_trends(train_df, dataset_id)
        self._plot_correlation_matrix(train_df, dataset_id)
        self._plot_operating_conditions(train_df, dataset_id)

    def _plot_sensor_distributions(self, df, dataset_id):
        """
        Plot distributions of sensor readings
        """
        fig = make_subplots(rows=7, cols=3, subplot_titles=self.sensor_cols)
        row, col = 1, 1

        for sensor in self.sensor_cols:
            fig.add_trace(
                go.Histogram(x=df[sensor], name=sensor, showlegend=False),
                row=row, col=col
            )

            col += 1
            if col > 3:
                col = 1
                row += 1

        fig.update_layout(height=1500, width=1000,
                         title_text=f"Sensor Distributions - Dataset {dataset_id}")
        fig.show()

    def _plot_sensor_trends(self, df, dataset_id):
        """
        Plot sensor trends for first engine
        """
        engine_1 = df[df['unit'] == 1]

        fig = make_subplots(rows=7, cols=3, subplot_titles=self.sensor_cols)
        row, col = 1, 1

        for sensor in self.sensor_cols:
            fig.add_trace(
                go.Scatter(x=engine_1['cycle'], y=engine_1[sensor],
                          name=sensor, showlegend=False),
                row=row, col=col
            )

            col += 1
            if col > 3:
                col = 1
                row += 1

        fig.update_layout(height=1500, width=1000,
                         title_text=f"Sensor Trends (Engine 1) - Dataset {dataset_id}")
        fig.show()

    def _plot_correlation_matrix(self, df, dataset_id):
        """
        Plot correlation matrix for sensors
        """
        corr_matrix = df[self.sensor_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=self.sensor_cols,
            y=self.sensor_cols,
            colorscale='RdBu',
            zmid=0
        ))

        fig.update_layout(
            title=f'Sensor Correlation Matrix - Dataset {dataset_id}',
            height=800,
            width=800
        )
        fig.show()

    def _plot_operating_conditions(self, df, dataset_id):
        """
        Plot operating conditions distribution
        """
        fig = make_subplots(rows=1, cols=3, subplot_titles=self.setting_cols)

        for i, setting in enumerate(self.setting_cols, 1):
            fig.add_trace(
                go.Histogram(x=df[setting], name=setting),
                row=1, col=i
            )

        fig.update_layout(
            height=400, width=1200,
            title_text=f"Operating Conditions Distribution - Dataset {dataset_id}"
        )
        fig.show()

    # Part 2: Data Preprocessing and Feature Engineering

    def preprocess_data(self, dataset_id='FD001'):
        """
        Comprehensive data preprocessing including RUL calculation and normalization
        """
        print(f"\nPreprocessing Dataset {dataset_id}")

        train_df = self.datasets[dataset_id]['train'].copy()
        test_df = self.datasets[dataset_id]['test'].copy()
        rul_df = self.datasets[dataset_id]['rul'].copy()

        # Calculate RUL for training data
        rul = pd.DataFrame(train_df.groupby('unit')['cycle'].max()).reset_index()
        rul.columns = ['unit', 'max_cycle']
        train_df = train_df.merge(rul, on=['unit'], how='left')
        train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
        train_df.drop('max_cycle', axis=1, inplace=True)

        # Add RUL for test data
        max_cycles_test = test_df.groupby('unit')['cycle'].max().reset_index()
        max_cycles_test.columns = ['unit', 'max_cycle']
        test_df = test_df.merge(max_cycles_test, on=['unit'], how='left')

        # Combine true RUL values with test data
        test_rul_true = pd.DataFrame()
        test_rul_true['unit'] = range(1, len(rul_df) + 1)
        test_rul_true['RUL_true'] = rul_df['RUL'].values
        test_df = test_df.merge(test_rul_true, on=['unit'], how='left')
        test_df['RUL'] = test_df['RUL_true'] + (test_df['max_cycle'] - test_df['cycle'])
        test_df.drop(['max_cycle', 'RUL_true'], axis=1, inplace=True)

        # Store processed dataframes
        self.datasets[dataset_id]['train_processed'] = train_df
        self.datasets[dataset_id]['test_processed'] = test_df

        print("Basic preprocessing completed")
        return train_df, test_df

    def engineer_features(self, dataset_id='FD001'):
        """
        Comprehensive feature engineering
        """
        print(f"\nEngineering features for Dataset {dataset_id}")

        train_df = self.datasets[dataset_id]['train_processed'].copy()
        test_df = self.datasets[dataset_id]['test_processed'].copy()

        # Create window statistics features
        window_sizes = [5, 10, 20]

        for df in [train_df, test_df]:
            # Moving averages
            for sensor in self.sensor_cols:
                for window in window_sizes:
                    df[f'{sensor}_ma_{window}'] = df.groupby('unit')[sensor].rolling(
                        window=window, min_periods=1).mean().reset_index(0, drop=True)

            # Moving standard deviations
            for sensor in self.sensor_cols:
                for window in window_sizes:
                    df[f'{sensor}_std_{window}'] = df.groupby('unit')[sensor].rolling(
                        window=window, min_periods=1).std().reset_index(0, drop=True)

            # Rate of change features
            for sensor in self.sensor_cols:
                # First order difference
                df[f'{sensor}_rate'] = df.groupby('unit')[sensor].diff()

                # Second order difference
                df[f'{sensor}_rate2'] = df.groupby('unit')[f'{sensor}_rate'].diff()

                # Percentage change
                df[f'{sensor}_pct_change'] = df.groupby('unit')[sensor].pct_change()

            # Cumulative statistics
            for sensor in self.sensor_cols:
                df[f'{sensor}_cumsum'] = df.groupby('unit')[sensor].cumsum()
                df[f'{sensor}_cummax'] = df.groupby('unit')[sensor].cummax()
                df[f'{sensor}_cummin'] = df.groupby('unit')[sensor].cummin()

            # Interaction features between sensors
            important_sensors = ['sensor2', 'sensor7', 'sensor12', 'sensor15']
            for i in range(len(important_sensors)):
                for j in range(i+1, len(important_sensors)):
                    s1, s2 = important_sensors[i], important_sensors[j]
                    df[f'{s1}_{s2}_ratio'] = df[s1] / df[s2]
                    df[f'{s1}_{s2}_diff'] = df[s1] - df[s2]

            # Time-based features
            df['cycle_normalized'] = df.groupby('unit')['cycle'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min()))

            # Operating condition interaction features
            for i in range(1, 4):
                for j in range(i+1, 4):
                    df[f'setting{i}{j}_interaction'] = df[f'setting{i}'] * df[f'setting{j}']

        # Fill NaN values
        train_df.fillna(0, inplace=True)
        test_df.fillna(0, inplace=True)

        # Store engineered dataframes
        self.datasets[dataset_id]['train_engineered'] = train_df
        self.datasets[dataset_id]['test_engineered'] = test_df

        print("Feature engineering completed")
        self._analyze_engineered_features(train_df, dataset_id)
        return train_df, test_df

    def _analyze_engineered_features(self, df, dataset_id):
        """
        Analyze the importance and relationships of engineered features
        """
        # Calculate feature correlations with RUL
        correlations = df.corr()['RUL'].sort_values(ascending=False)

        print("\nTop 10 features most correlated with RUL:")
        print(correlations[:10])

        # Visualize top feature correlations
        plt.figure(figsize=(12, 6))
        correlations[:20].plot(kind='bar')
        plt.title(f'Top 20 Feature Correlations with RUL - Dataset {dataset_id}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Feature importance using Random Forest
        feature_cols = [col for col in df.columns if col not in ['unit', 'RUL']]
        X = df[feature_cols]
        y = df['RUL']

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance[:20])
        plt.title(f'Top 20 Feature Importance - Dataset {dataset_id}')
        plt.tight_layout()
        plt.show()

    def normalize_data(self, dataset_id='FD001'):
        """
        Normalize the engineered features
        """
        train_df = self.datasets[dataset_id]['train_engineered'].copy()
        test_df = self.datasets[dataset_id]['test_engineered'].copy()

        # Identify features to normalize (exclude unit, cycle, and RUL)
        features_to_normalize = [col for col in train_df.columns
                               if col not in ['unit', 'cycle', 'RUL']]

        # Initialize scaler
        self.scalers[dataset_id] = StandardScaler()

        # Normalize training data
        train_df[features_to_normalize] = self.scalers[dataset_id].fit_transform(
            train_df[features_to_normalize])

        # Normalize test data
        test_df[features_to_normalize] = self.scalers[dataset_id].transform(
            test_df[features_to_normalize])

        # Store normalized dataframes
        self.datasets[dataset_id]['train_normalized'] = train_df
        self.datasets[dataset_id]['test_normalized'] = test_df

        print(f"\nData normalization completed for Dataset {dataset_id}")
        return train_df, test_df

    # Part 3: Advanced Anomaly Detection
    def detect_anomalies(self, dataset_id='FD001'):
        """
        Comprehensive anomaly detection using multiple approaches
        """
        print(f"\nPerforming anomaly detection for Dataset {dataset_id}")

        train_df = self.datasets[dataset_id]['train_normalized'].copy()
        test_df = self.datasets[dataset_id]['test_normalized'].copy()

        # Store anomaly detection results
        self.datasets[dataset_id]['anomaly_results'] = {}

        # 1. Statistical-based anomaly detection
        self._statistical_anomaly_detection(train_df, dataset_id)

        # 2. Isolation Forest-based anomaly detection
        self._isolation_forest_detection(train_df, dataset_id)

        # 3. LSTM Autoencoder-based anomaly detection
        self._autoencoder_anomaly_detection(train_df, test_df, dataset_id)

        # 4. Combine and analyze anomaly detection results
        self._analyze_anomaly_results(dataset_id)

        return self.datasets[dataset_id]['anomaly_results']

    def _statistical_anomaly_detection(self, df, dataset_id):
        """
        Statistical approach to anomaly detection using Z-score and IQR
        """
        results = pd.DataFrame()

        # Z-score based detection
        for sensor in self.sensor_cols:
            z_scores = np.abs(stats.zscore(df[sensor]))
            results[f'{sensor}_zscore_anomaly'] = (z_scores > 3).astype(int)

        # IQR based detection
        for sensor in self.sensor_cols:
            Q1 = df[sensor].quantile(0.25)
            Q3 = df[sensor].quantile(0.75)
            IQR = Q3 - Q1
            results[f'{sensor}_iqr_anomaly'] = ((df[sensor] < (Q1 - 1.5 * IQR)) |
                                               (df[sensor] > (Q3 + 1.5 * IQR))).astype(int)

        # Combine results
        results['statistical_anomaly'] = (results.sum(axis=1) > 0).astype(int)
        self.datasets[dataset_id]['anomaly_results']['statistical'] = results

        # Visualize statistical anomalies
        self._plot_statistical_anomalies(df, results, dataset_id)

    def _isolation_forest_detection(self, df, dataset_id):
        """
        Isolation Forest based anomaly detection
        """
        # Initialize and fit Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(df[self.sensor_cols])

        # Convert predictions to binary (1: normal, 0: anomaly)
        anomalies = pd.Series(anomalies).map({1: 0, -1: 1})

        self.datasets[dataset_id]['anomaly_results']['isolation_forest'] = anomalies

        # Visualize Isolation Forest results
        self._plot_isolation_forest_anomalies(df, anomalies, dataset_id)

    def _autoencoder_anomaly_detection(self, train_df, test_df, dataset_id):
        """
        LSTM Autoencoder based anomaly detection
        """
        # Prepare sequences for autoencoder
        sequence_length = 30
        X_train = self._prepare_sequences(train_df[self.sensor_cols], sequence_length)

        # Build and train autoencoder
        autoencoder = self._build_autoencoder(X_train.shape[2], sequence_length)

        history = autoencoder.fit(
            X_train, X_train,
            epochs=10,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )

        # Get reconstruction error
        train_pred = autoencoder.predict(X_train)
        reconstruction_error = np.mean(np.abs(X_train - train_pred), axis=(1,2))

        # Define threshold for anomaly detection
        threshold = np.percentile(reconstruction_error, 95)

        # Detect anomalies
        anomalies = (reconstruction_error > threshold).astype(int)

        # Store results
        self.datasets[dataset_id]['anomaly_results']['autoencoder'] = anomalies

        # Visualize autoencoder results
        self._plot_autoencoder_anomalies(train_df, anomalies, reconstruction_error, threshold, dataset_id)

    def _build_autoencoder(self, n_features, sequence_length):
        """
        Build LSTM Autoencoder model
        """
        model = Sequential([
            # Encoder
            LSTM(64, activation='relu', input_shape=(sequence_length, n_features),
                 return_sequences=True),
            LSTM(32, activation='relu', return_sequences=False),

            # Decoder
            RepeatVector(sequence_length),
            LSTM(32, activation='relu', return_sequences=True),
            LSTM(64, activation='relu', return_sequences=True),
            TimeDistributed(Dense(n_features))
        ])

        model.compile(optimizer='adam', loss='mse')
        return model

    def _prepare_sequences(self, data, sequence_length):
        """
        Prepare sequences for LSTM autoencoder
        """
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data.iloc[i:i+sequence_length].values)
        return np.array(sequences)

    def _plot_statistical_anomalies(self, df, results, dataset_id):
        """
        Visualize statistical anomalies
        """
        fig = make_subplots(rows=3, cols=2, subplot_titles=self.sensor_cols[:6])

        for i, sensor in enumerate(self.sensor_cols[:6], 1):
            row = (i-1) // 2 + 1
            col = (i-1) % 2 + 1

            # Normal points
            fig.add_trace(
                go.Scatter(
                    x=df[results[f'{sensor}_zscore_anomaly'] == 0].index,
                    y=df[results[f'{sensor}_zscore_anomaly'] == 0][sensor],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='blue', size=2),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Anomaly points
            fig.add_trace(
                go.Scatter(
                    x=df[results[f'{sensor}_zscore_anomaly'] == 1].index,
                    y=df[results[f'{sensor}_zscore_anomaly'] == 1][sensor],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=4),
                    showlegend=False
                ),
                row=row, col=col
            )

        fig.update_layout(height=800, width=1000,
                         title_text=f"Statistical Anomalies - Dataset {dataset_id}")
        fig.show()

    def _plot_isolation_forest_anomalies(self, df, anomalies, dataset_id):
        """
        Visualize Isolation Forest anomalies
        """
        fig = make_subplots(rows=3, cols=2, subplot_titles=self.sensor_cols[:6])

        for i, sensor in enumerate(self.sensor_cols[:6], 1):
            row = (i-1) // 2 + 1
            col = (i-1) % 2 + 1

            # Normal points
            fig.add_trace(
                go.Scatter(
                    x=df[anomalies == 0].index,
                    y=df[anomalies == 0][sensor],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='blue', size=2),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Anomaly points
            fig.add_trace(
                go.Scatter(
                    x=df[anomalies == 1].index,
                    y=df[anomalies == 1][sensor],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=4),
                    showlegend=False
                ),
                row=row, col=col
            )

        fig.update_layout(height=800, width=1000,
                         title_text=f"Isolation Forest Anomalies - Dataset {dataset_id}")
        fig.show()

    def _plot_autoencoder_anomalies(self, df, anomalies, reconstruction_error, threshold, dataset_id):
        """
        Visualize Autoencoder anomalies
        """
        # Plot reconstruction error distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=reconstruction_error, name='Reconstruction Error'))
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                     annotation_text="Threshold")
        fig.update_layout(title_text=f"Reconstruction Error Distribution - Dataset {dataset_id}")
        fig.show()

        # Plot anomalies for key sensors
        fig = make_subplots(rows=3, cols=2, subplot_titles=self.sensor_cols[:6])

        for i, sensor in enumerate(self.sensor_cols[:6], 1):
            row = (i-1) // 2 + 1
            col = (i-1) % 2 + 1

            # Normal points
            fig.add_trace(
                go.Scatter(
                    x=df[anomalies == 0].index,
                    y=df[anomalies == 0][sensor],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='blue', size=2),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Anomaly points
            fig.add_trace(
                go.Scatter(
                    x=df[anomalies == 1].index,
                    y=df[anomalies == 1][sensor],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=4),
                    showlegend=False
                ),
                row=row, col=col
            )

        fig.update_layout(height=800, width=1000,
                         title_text=f"Autoencoder Anomalies - Dataset {dataset_id}")
        fig.show()

    def _analyze_anomaly_results(self, dataset_id):
        """
        Analyze and compare results from different anomaly detection methods
        """
        results = self.datasets[dataset_id]['anomaly_results']

        # Compare detection rates
        detection_rates = {
            'Statistical': results['statistical']['statistical_anomaly'].mean(),
            'Isolation Forest': results['isolation_forest'].mean(),
            'Autoencoder': results['autoencoder'].mean()
        }

        # Plot comparison
        fig = go.Figure([go.Bar(x=list(detection_rates.keys()),
                              y=list(detection_rates.values()))])
        fig.update_layout(title_text=f"Anomaly Detection Rates Comparison - Dataset {dataset_id}",
                         xaxis_title="Method",
                         yaxis_title="Detection Rate")
        fig.show()

        # Calculate agreement between methods
        print("\nAgreement between methods:")
        methods = list(results.keys())
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                agreement = np.mean(results[methods[i]] == results[methods[j]])
                print(f"{methods[i]} vs {methods[j]}: {agreement:.2%} agreement")

    # Part 4: Advanced Model Building and Evaluation

    def build_advanced_model(self, dataset_id='FD001'):
        """
        Build and train advanced LSTM model for RUL prediction
        """
        print(f"\nBuilding advanced model for Dataset {dataset_id}")

        train_df = self.datasets[dataset_id]['train_normalized']
        test_df = self.datasets[dataset_id]['test_normalized']

        # Prepare sequences for training
        sequence_length = 50
        features = self._get_feature_columns(train_df)

        X_train, y_train = self._prepare_model_sequences(train_df, features, sequence_length)
        X_test, y_test = self._prepare_model_sequences(test_df, features, sequence_length)

        # Build and train model
        model = self._build_advanced_lstm(X_train.shape[2], sequence_length)

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        # Store model and training history
        self.models[dataset_id] = {
            'model': model,
            'history': history,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }

        # Analyze training results
        self._analyze_training_results(dataset_id)

        return model, history

    def _build_advanced_lstm(self, n_features, sequence_length):
        """
        Build advanced LSTM model architecture
        """
        model = Sequential([
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(128, return_sequences=True),
                         input_shape=(sequence_length, n_features)),
            Dropout(0.3),

            # Second Bidirectional LSTM layer
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),

            # Third Bidirectional LSTM layer
            Bidirectional(LSTM(32)),
            Dropout(0.3),

            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        # Compile model with custom loss function
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self._custom_loss
        )

        return model

    def _custom_loss(self, y_true, y_pred):
        """
        Custom asymmetric loss function for RUL prediction
        Penalizes underestimation more than overestimation
        """
        error = y_true - y_pred
        underestimation_penalty = tf.maximum(0., error) * 1.3
        overestimation_penalty = tf.maximum(0., -error)
        return tf.reduce_mean(tf.square(underestimation_penalty + overestimation_penalty))

    def _get_feature_columns(self, df):
        """
        Get relevant feature columns for model training
        """
        exclude_cols = ['unit', 'cycle', 'RUL']
        return [col for col in df.columns if col not in exclude_cols]

    def _prepare_model_sequences(self, df, features, sequence_length):
        """
        Prepare sequences for model training with advanced windowing
        """
        sequences = []
        targets = []

        for unit in df['unit'].unique():
            unit_data = df[df['unit'] == unit]

            # Create sequences
            for i in range(len(unit_data) - sequence_length + 1):
                sequences.append(unit_data[features].iloc[i:i+sequence_length].values)
                targets.append(unit_data['RUL'].iloc[i+sequence_length-1])

        return np.array(sequences), np.array(targets)

    def _analyze_training_results(self, dataset_id):
        """
        Comprehensive analysis of model training results
        """
        history = self.models[dataset_id]['history']
        model = self.models[dataset_id]['model']
        X_train = self.models[dataset_id]['X_train']
        y_train = self.models[dataset_id]['y_train']
        X_test = self.models[dataset_id]['X_test']
        y_test = self.models[dataset_id]['y_test']

        # Plot training history
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(y=history.history['loss'], name='Training Loss')
        )
        fig.add_trace(
            go.Scatter(y=history.history['val_loss'], name='Validation Loss')
        )
        fig.update_layout(title=f'Training History - Dataset {dataset_id}',
                         xaxis_title='Epoch',
                         yaxis_title='Loss')
        fig.show()

        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        print("\nModel Performance Metrics:")
        print(f"Training RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Training R2 Score: {train_r2:.2f}")
        print(f"Test R2 Score: {test_r2:.2f}")

        # Plot predictions vs actual
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Training Predictions', 'Test Predictions'))

        # Training predictions
        fig.add_trace(
            go.Scatter(x=y_train, y=train_pred.flatten(),
                      mode='markers',
                      name='Training'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[y_train.min(), y_train.max()],
                      y=[y_train.min(), y_train.max()],
                      mode='lines',
                      name='Perfect Prediction',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )

        # Test predictions
        fig.add_trace(
            go.Scatter(x=y_test, y=test_pred.flatten(),
                      mode='markers',
                      name='Test'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=[y_test.min(), y_test.max()],
                      y=[y_test.min(), y_test.max()],
                      mode='lines',
                      name='Perfect Prediction',
                      line=dict(color='red', dash='dash')),
            row=1, col=2
        )

        fig.update_layout(height=500, width=1000,
                         title_text=f"Prediction Analysis - Dataset {dataset_id}")
        fig.show()

        # Error distribution analysis
        train_errors = y_train - train_pred.flatten()
        test_errors = y_test - test_pred.flatten()

        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Training Error Distribution',
                                         'Test Error Distribution'))

        fig.add_trace(
            go.Histogram(x=train_errors, name='Training Errors'),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=test_errors, name='Test Errors'),
            row=1, col=2
        )

        fig.update_layout(height=400, width=1000,
                         title_text=f"Error Distribution Analysis - Dataset {dataset_id}")
        fig.show()

    def evaluate_all_datasets(self):
        """
        Evaluate model performance across all datasets
        """
        results = {}

        for dataset_id in ['FD001', 'FD002', 'FD003', 'FD004']:
            if dataset_id in self.models:
                model = self.models[dataset_id]['model']
                X_test = self.models[dataset_id]['X_test']
                y_test = self.models[dataset_id]['y_test']

                # Make predictions
                predictions = model.predict(X_test)

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                r2 = r2_score(y_test, predictions)

                results[dataset_id] = {
                    'RMSE': rmse,
                    'R2': r2
                }

        # Plot comparison
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('RMSE Comparison', 'R2 Score Comparison'))

        fig.add_trace(
            go.Bar(x=list(results.keys()),
                  y=[results[k]['RMSE'] for k in results.keys()],
                  name='RMSE'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=list(results.keys()),
                  y=[results[k]['R2'] for k in results.keys()],
                  name='R2 Score'),
            row=1, col=2
        )

        fig.update_layout(height=400, width=1000,
                         title_text="Model Performance Comparison Across Datasets")
        fig.show()

        return results

    def process_dataset(self, dataset_id):
        """
        Process a single dataset through all steps
        """
        try:
            # Load data
            if not self.load_data(dataset_id):
                return False

            # Perform analysis
            if not self.dataset_analysis(dataset_id):
                return False

            # Preprocess data
            train_df, test_df = self.preprocess_data(dataset_id)
            self.datasets[dataset_id]['train_processed'] = train_df
            self.datasets[dataset_id]['test_processed'] = test_df

            # Engineer features
            train_df, test_df = self.engineer_features(dataset_id)
            self.datasets[dataset_id]['train_engineered'] = train_df
            self.datasets[dataset_id]['test_engineered'] = test_df

            # Normalize data
            train_df, test_df = self.normalize_data(dataset_id)
            self.datasets[dataset_id]['train_normalized'] = train_df
            self.datasets[dataset_id]['test_normalized'] = test_df

            return True

        except Exception as e:
            print(f"Error processing dataset {dataset_id}: {str(e)}")
            return False


    # Part 5: Main Execution Flow and Utility Functions
    def run_complete_analysis(self, datasets=['FD001', 'FD002', 'FD003', 'FD004']):
        """
        Run complete analysis pipeline for specified datasets
        """
        overall_results = {}
        processed_datasets = []

        for dataset_id in datasets:
            print(f"\n{'='*50}")
            print(f"Processing Dataset {dataset_id}")
            print(f"{'='*50}")

            try:
                # Process dataset
                if not self.process_dataset(dataset_id):
                    continue

                # Detect anomalies
                anomaly_results = self.detect_anomalies(dataset_id)
                self.datasets[dataset_id]['anomaly_results'] = anomaly_results

                # Build and evaluate model
                model, history = self.build_advanced_model(dataset_id)

                # Store results
                overall_results[dataset_id] = {
                    'model': model,
                    'history': history,
                    'anomalies': anomaly_results
                }

                processed_datasets.append(dataset_id)

            except Exception as e:
                print(f"Error processing dataset {dataset_id}: {str(e)}")
                continue

        # Generate reports only for successfully processed datasets
        for dataset_id in processed_datasets:
            self.generate_report(dataset_id)

        # Compare results across datasets
        if processed_datasets:
            self.compare_dataset_results(overall_results)

        return overall_results

    def compare_dataset_results(self, overall_results):
        """
        Compare and visualize results across different datasets
        """
        # Prepare comparison metrics
        comparison = {
            'dataset': [],
            'train_samples': [],
            'test_samples': [],
            'anomaly_rate': [],
            'final_loss': [],
            'training_time': []
        }

        for dataset_id, results in overall_results.items():
            comparison['dataset'].append(dataset_id)
            comparison['train_samples'].append(len(self.datasets[dataset_id]['train_normalized']))
            comparison['test_samples'].append(len(self.datasets[dataset_id]['test_normalized']))
            comparison['anomaly_rate'].append(
                np.mean(self.datasets[dataset_id]['anomaly_results']['isolation_forest']))
            comparison['final_loss'].append(results['history'].history['loss'][-1])
            comparison['training_time'].append(len(results['history'].history['loss']))

        # Create comparison visualizations
        self._plot_dataset_comparisons(comparison)

    def _plot_dataset_comparisons(self, comparison):
        """
        Create comparative visualizations across datasets
        """
        # Create subplots for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sample Sizes', 'Anomaly Rates',
                          'Final Loss', 'Training Duration')
        )

        # Sample sizes
        fig.add_trace(
            go.Bar(name='Train Samples',
                  x=comparison['dataset'],
                  y=comparison['train_samples']),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Test Samples',
                  x=comparison['dataset'],
                  y=comparison['test_samples']),
            row=1, col=1
        )

        # Anomaly rates
        fig.add_trace(
            go.Bar(x=comparison['dataset'],
                  y=comparison['anomaly_rate']),
            row=1, col=2
        )

        # Final loss
        fig.add_trace(
            go.Bar(x=comparison['dataset'],
                  y=comparison['final_loss']),
            row=2, col=1
        )

        # Training duration
        fig.add_trace(
            go.Bar(x=comparison['dataset'],
                  y=comparison['training_time']),
            row=2, col=2
        )

        fig.update_layout(height=800, width=1200,
                         title_text="Dataset Comparison Analysis")
        fig.show()

    def generate_report(self, dataset_id):
        """
        Generate comprehensive analysis report for a dataset
        """
        if dataset_id not in self.datasets:
            print(f"Dataset {dataset_id} not found")
            return None

        if 'train_normalized' not in self.datasets[dataset_id]:
            print(f"Normalized data not found for dataset {dataset_id}")
            return None

        report = {
            'dataset_info': self._get_dataset_info(dataset_id),
            'feature_analysis': self._get_feature_analysis(dataset_id),
            'anomaly_analysis': self._get_anomaly_analysis(dataset_id),
            'model_performance': self._get_model_performance(dataset_id)
        }

        self._display_report(report, dataset_id)

        return report

    def _get_dataset_info(self, dataset_id):
        """
        Get basic dataset information
        """
        try:
            train_df = self.datasets[dataset_id]['train_normalized']
            test_df = self.datasets[dataset_id]['test_normalized']

            return {
                'train_shape': train_df.shape,
                'test_shape': test_df.shape,
                'n_units_train': train_df['unit'].nunique(),
                'n_units_test': test_df['unit'].nunique(),
                'avg_cycles_train': train_df.groupby('unit')['cycle'].max().mean(),
                'avg_cycles_test': test_df.groupby('unit')['cycle'].max().mean()
            }
        except KeyError as e:
            print(f"Error getting dataset info for {dataset_id}: {str(e)}")
            return None

    def _get_feature_analysis(self, dataset_id):
        """
        Get feature analysis results
        """
        train_df = self.datasets[dataset_id]['train_normalized']

        return {
            'n_features': len(self._get_feature_columns(train_df)),
            'top_correlations': train_df.corr()['RUL'].sort_values(ascending=False)[:10],
            'feature_stats': train_df[self.sensor_cols].describe()
        }

    def _get_anomaly_analysis(self, dataset_id):
        """
        Get anomaly detection results
        """
        anomaly_results = self.datasets[dataset_id]['anomaly_results']

        return {
            'statistical_rate': np.mean(anomaly_results['statistical']['statistical_anomaly']),
            'isolation_forest_rate': np.mean(anomaly_results['isolation_forest']),
            'autoencoder_rate': np.mean(anomaly_results['autoencoder'])
        }

    def _get_model_performance(self, dataset_id):
        """
        Get model performance metrics
        """
        model_results = self.models[dataset_id]
        y_test = model_results['y_test']
        test_pred = model_results['model'].predict(model_results['X_test'])

        return {
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'test_r2': r2_score(y_test, test_pred),
            'final_loss': model_results['history'].history['loss'][-1],
            'n_epochs': len(model_results['history'].history['loss'])
        }

    def _display_report(self, report, dataset_id):
        """
        Display comprehensive analysis report
        """
        print(f"\n{'='*50}")
        print(f"Analysis Report for Dataset {dataset_id}")
        print(f"{'='*50}")

        # Dataset Information
        print("\nDataset Information:")
        print(f"Training samples: {report['dataset_info']['train_shape'][0]}")
        print(f"Test samples: {report['dataset_info']['test_shape'][0]}")
        print(f"Number of training units: {report['dataset_info']['n_units_train']}")
        print(f"Average cycles per unit (train): {report['dataset_info']['avg_cycles_train']:.2f}")

        # Feature Analysis
        print("\nFeature Analysis:")
        print(f"Number of features: {report['feature_analysis']['n_features']}")
        print("\nTop feature correlations with RUL:")
        print(report['feature_analysis']['top_correlations'])

        # Anomaly Analysis
        print("\nAnomaly Detection Results:")
        print(f"Statistical method detection rate: {report['anomaly_analysis']['statistical_rate']:.2%}")
        print(f"Isolation Forest detection rate: {report['anomaly_analysis']['isolation_forest_rate']:.2%}")
        print(f"Autoencoder detection rate: {report['anomaly_analysis']['autoencoder_rate']:.2%}")

        # Model Performance
        print("\nModel Performance:")
        print(f"Test RMSE: {report['model_performance']['test_rmse']:.2f}")
        print(f"Test R2 Score: {report['model_performance']['test_r2']:.2f}")
        print(f"Final training loss: {report['model_performance']['final_loss']:.4f}")
        print(f"Training epochs: {report['model_performance']['n_epochs']}")

# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TurbofanAnalysis()

    # Run complete analysis for all datasets
    results = analyzer.run_complete_analysis()

    # Generate reports for each dataset
    for dataset_id in ['FD001', 'FD002', 'FD003', 'FD004']:
        analyzer.generate_report(dataset_id)

    # Compare results across datasets
    analyzer.evaluate_all_datasets()