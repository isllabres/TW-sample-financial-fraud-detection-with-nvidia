"""
gnn_data_generation.py
---------------------
Module for generating GNN graph, node features, and node labels from cleaned TabFormer data.
"""

from typing import Dict, Any, Optional, List
import os
import json
import numpy as np
import pandas as pd
from scipy.linalg import block_diag

from config import DataConfig, default_config


def create_feature_mask(columns):
    # Dictionary to store mapping from original column to mask value
    mask_mapping = {}
    mask_values = []
    current_mask = 0

    for col in columns:
        # For encoded columns, assume the base is before the underscore
        if "_" in col:
            base_feature = col.split("_")[0]
        else:
            base_feature = col  # For non-encoded columns, use the column name directly

        # Assign a new mask value if this base feature hasn't been seen before
        if base_feature not in mask_mapping:
            mask_mapping[base_feature] = current_mask
            current_mask += 1

        # Append the mask value for this column
        mask_values.append(mask_mapping[base_feature])

    # Convert list to numpy array for further processing if needed
    feature_mask = np.array(mask_values)

    return mask_mapping, feature_mask


def generate_gnn_graph_data(
    cleaned_data_bundle: Dict[str, Any],
    output_dir: Optional[str] = None,
    data_split_year: Optional[int] = None,
    xgb_transformer=None,
    columns_of_transformed_data: Optional[List[str]] = None,
    config: Optional[DataConfig] = None
) -> Dict[str, Any]:
    """Generate GNN graph data from cleaned data.
    
    Args:
        cleaned_data_bundle: Output from load_and_clean_tabformer
        output_dir: Directory to save graph data. If None, uses config.output_dir
        data_split_year: Year to split data. If None, uses config.test_split_year
        xgb_transformer: Transformer from XGBoost feature generation
        columns_of_transformed_data: Feature columns from XGBoost
        config: Configuration object. If None, uses config from cleaned_data_bundle or default
        
    Returns:
        Dictionary containing graph data and metadata
    """
    if config is None:
        config = cleaned_data_bundle.get('config', default_config)
        
    if output_dir is None:
        output_dir = os.path.join(config.output_dir, "gnn")
        os.makedirs(output_dir, exist_ok=True)
        
    if data_split_year is None:
        data_split_year = config.test_split_year
        
    # Get transformer and feature columns from XGBoost if not provided
    if xgb_transformer is None:
        xgb_transformer = cleaned_data_bundle.get('xgb_transformer')
    if columns_of_transformed_data is None:
        columns_of_transformed_data = cleaned_data_bundle.get('xgb_feature_columns', [])
    """
    Generate GNN graph, node features, and node labels from cleaned TabFormer data.

    Args:
        cleaned_data_bundle (dict): Output from load_and_clean_tabformer.
        output_dir (str): Directory to write GNN graph files.
        data_split_year (int): Year to split data into train/validation/test splits.
    """
    data = cleaned_data_bundle["data"]
    id_transformer = cleaned_data_bundle["id_transformer"]
    columns_of_transformed_id_data = cleaned_data_bundle[
        "columns_of_transformed_id_data"
    ]
    # Split data using config
    training_idx = data[config.COL_YEAR] < data_split_year
    validation_idx = data[config.COL_YEAR] == data_split_year
    test_idx = data[config.COL_YEAR] > data_split_year

    # For GNN, use train+val for main graph, test for test graph
    def process_gnn_split(sub_data, out_dir):
        sub_data = sub_data.copy().reset_index(drop=True)
        sub_data[config.COL_TRANSACTION_ID] = sub_data.index
        merchant_name_to_id = dict(
            zip(
                sub_data[config.COL_MERCHANT].unique(),
                np.arange(len(sub_data[config.COL_MERCHANT].unique())),
            )
        )
        sub_data[config.COL_MERCHANT_ID] = sub_data[config.COL_MERCHANT].map(merchant_name_to_id)
        id_to_consecutive_id = dict(
            zip(
                sub_data[config.COL_CARD].unique(), 
                np.arange(len(sub_data[config.COL_CARD].unique()))
            )
        )
        sub_data[config.COL_USER_ID] = sub_data[config.COL_CARD].map(id_to_consecutive_id)
        NR_USERS = sub_data[config.COL_USER_ID].max() + 1
        NR_MXS = sub_data[config.COL_MERCHANT_ID].max() + 1
        NR_TXS = sub_data[config.COL_TRANSACTION_ID].max() + 1
        # Check the the transaction, merchant and user ids are consecutive
        id_range = sub_data[config.COL_TRANSACTION_ID].min(), sub_data[config.COL_TRANSACTION_ID].max()
        print(f"Transaction ID range {id_range}")
        id_range = sub_data[config.COL_MERCHANT_ID].min(), sub_data[config.COL_MERCHANT_ID].max()
        print(f"Merchant ID range {id_range}")
        id_range = sub_data[config.COL_USER_ID].min(), sub_data[config.COL_USER_ID].max()
        print(f"User ID range {id_range}")
        # Edges
        U_2_T = pd.DataFrame(
            {
                config.COL_GRAPH_SRC: sub_data[config.COL_USER_ID],
                config.COL_GRAPH_DST: sub_data[config.COL_TRANSACTION_ID] + NR_USERS + NR_MXS,
            }
        )
        T_2_M = pd.DataFrame(
            {
                config.COL_GRAPH_SRC: sub_data[config.COL_TRANSACTION_ID] + NR_USERS + NR_MXS,
                config.COL_GRAPH_DST: sub_data[config.COL_MERCHANT_ID] + NR_USERS,
            }
        )
        T_2_U = pd.DataFrame(
            {
                config.COL_GRAPH_SRC: sub_data[config.COL_TRANSACTION_ID] + NR_USERS + NR_MXS,
                config.COL_GRAPH_DST: sub_data[config.COL_USER_ID],
            }
        )
        M_2_T = pd.DataFrame(
            {
                config.COL_GRAPH_SRC: sub_data[config.COL_MERCHANT_ID] + NR_USERS,
                config.COL_GRAPH_DST: sub_data[config.COL_TRANSACTION_ID] + NR_USERS + NR_MXS,
            }
        )
        edges = pd.concat([U_2_T, T_2_M, T_2_U, M_2_T], ignore_index=True)
        os.makedirs(os.path.join(out_dir, "edges"), exist_ok=True)
        edges.to_csv(os.path.join(out_dir, "edges", "node_to_node.csv"), index=False)
        # Node features
        data_merchant = sub_data[[config.COL_MERCHANT, config.COL_MCC, config.COL_CARD]].drop_duplicates(
            subset=[config.COL_MERCHANT]
        )
        data_merchant[config.COL_MERCHANT_ID] = data_merchant[config.COL_MERCHANT].map(
            merchant_name_to_id
        )
        data_merchant_sorted = data_merchant.sort_values(by=config.COL_MERCHANT_ID)
        data_user = sub_data[[config.COL_MERCHANT, config.COL_MCC, config.COL_CARD]].drop_duplicates(
            subset=[config.COL_CARD]
        )
        data_user[config.COL_USER_ID] = data_user[config.COL_CARD].map(id_to_consecutive_id)
        data_user_sorted = data_user.sort_values(by=config.COL_USER_ID)
        user_feature_columns = [
            c for c in columns_of_transformed_id_data if c.startswith("Card")
        ]
        mx_feature_columns = [
            c for c in columns_of_transformed_id_data if not c.startswith("Card")
        ]
        preprocessed_merchant_data = pd.DataFrame(
            id_transformer.transform(data_merchant_sorted[config.MERCHANT_AND_USER_COLS]),
            columns=columns_of_transformed_id_data,
        )[mx_feature_columns]
        preprocessed_user_data = pd.DataFrame(
            id_transformer.transform(data_user_sorted[config.MERCHANT_AND_USER_COLS]),
            columns=columns_of_transformed_id_data,
        )[user_feature_columns]
        # Transaction features: use all XGBoost features (columns_of_transformed_data), not just ID columns
        numerical_predictors = [config.COL_AMOUNT]
        nominal_predictors = [
            config.COL_ERROR, config.COL_CARD, config.COL_CHIP, config.COL_CITY, config.COL_ZIP, config.COL_MCC, config.COL_MERCHANT
        ]
        predictors = list(set(nominal_predictors) - set(config.MERCHANT_AND_USER_COLS)) + numerical_predictors
        transaction_feature_df = pd.DataFrame(
            xgb_transformer.transform(sub_data[predictors]),
            columns=columns_of_transformed_data,
        )
        T = transaction_feature_df.values
        U = preprocessed_user_data.values
        M = preprocessed_merchant_data.values
        combined_cols = user_feature_columns + mx_feature_columns + list(columns_of_transformed_data)
        node_feature_df = pd.DataFrame(block_diag(U, M, T), columns=combined_cols)

        os.makedirs(os.path.join(out_dir, "nodes"), exist_ok=True)
        node_feature_df.to_csv(os.path.join(out_dir, "nodes", "node.csv"), index=False)
        # Node labels
        node_label_df = pd.DataFrame(
            np.zeros(len(node_feature_df), dtype=int), columns=[config.COL_FRAUD]
        )
        node_label_df.iloc[NR_USERS + NR_MXS : NR_USERS + NR_MXS + NR_TXS, 0] = (
            sub_data[config.COL_FRAUD].values
        )
        node_label_df.to_csv(
            os.path.join(out_dir, "nodes", "node_label.csv"), index=False
        )
        # Offset metadata
        with open(
            os.path.join(out_dir, "nodes", "offset_range_of_training_node.json"), "w"
        ) as json_file:
            json.dump(
                {
                    "start": int(NR_USERS + NR_MXS),
                    "end": int(NR_USERS + NR_MXS + NR_TXS),
                },
                json_file,
                indent=4,
            )
        return node_feature_df, node_label_df   

    # Train+val
    _, _ = process_gnn_split(data[training_idx | validation_idx], output_dir)
    # Test  
    node_feature_df_test, _ = process_gnn_split(data[test_idx], os.path.join(output_dir, "test_gnn"))

    return create_feature_mask(node_feature_df_test.columns) 