import sys
# sys.path.append("./data_preprocessing/")
from data_preprocessing.classification import role_classification
# from classification import role_classification
# from classification import role_classification
import pandas as pd
import venn

# data_path = "../data/contributions/global/"
def generate_developers_with_roles(data_path: str, choice: str):
    """
    This function generates a DataFrame containing information about common developers between the PyTorch and TensorFlow communities.
    Parameters:
    choice: 'common', 'all'

    Returns:
    df (pandas.DataFrame): A DataFrame containing the following columns:
        - developer: The name of the common developer.
        - pytorch_role: The role classification of the developer in the PyTorch community (either 'core' or 'peripheral').
        - tensorflow_role: The role classification of the developer in the TensorFlow community (either 'core' or 'peripheral').
    """
    # get role classficationos of each community of the whole version
    pytorch_community_data = pd.read_csv(f"{data_path}pytorch.csv")
    (
        _,
        pytorch_core_set,
        pytorch_peri_set,
        pytorch_dev_set,
        pytorch_active_user_set,
    ) = role_classification(pytorch_community_data)
    # make a set like {'dev1':'core', 'dev2':'peripheral', ...}
    pytorch_dev_set_dict = {}
    for dev in pytorch_dev_set:
        if dev in pytorch_core_set:
            pytorch_dev_set_dict[dev] = "core"
        else:
            pytorch_dev_set_dict[dev] = "peripheral"

    tensorflow_community_data = pd.read_csv(
        f"{data_path}tensorflow.csv"
    )
    (
        _,
        tensorflow_core_set,
        tensorflow_peri_set,
        tensorflow_dev_set,
        tensorflow_active_user_set,
    ) = role_classification(tensorflow_community_data)
    tensorflow_dev_set_dict = {}
    for dev in tensorflow_dev_set:
        if dev in tensorflow_core_set:
            tensorflow_dev_set_dict[dev] = "core"
        else:
            tensorflow_dev_set_dict[dev] = "peripheral"
    if choice == 'common':
        common_active_users = set(pytorch_active_user_set) & set(tensorflow_active_user_set)
        common_devs = set(pytorch_dev_set_dict.keys()) & set(tensorflow_dev_set_dict.keys())
        result = [
            [dev, pytorch_dev_set_dict[dev], tensorflow_dev_set_dict[dev]]
            for dev in common_devs]
        common_developers_data = pd.DataFrame(
            result, columns=["developer", "pytorch_role", "tensorflow_role"]
        )
        return common_developers_data, common_active_users
    elif choice == 'all':
        all_devs = set(pytorch_dev_set_dict.keys()) | set(tensorflow_dev_set_dict.keys())
        result = [
            [dev, pytorch_dev_set_dict.get(dev, None), tensorflow_dev_set_dict.get(dev, None)]
            for dev in all_devs]
        all_developers_data = pd.DataFrame(
            result, columns=["developer", "pytorch_role", "tensorflow_role"]
        )
        return all_developers_data
    else:
        print('Invalid choice. Please choose from "common" and "all".')
        return None        

def venn_graph(save_path:str, common_developers_data: pd.DataFrame):
    """
    generate venn graph of common developers(core and peripheral developers in either community)

    """
    
    core_in_pytorch = set(common_developers_data[common_developers_data['pytorch_role'] == 'core']['developer'])
    core_in_tensorflow = set(common_developers_data[common_developers_data['tensorflow_role'] == 'core']['developer'])
    peripheral_in_pytorch = set(common_developers_data[common_developers_data['pytorch_role'] == 'peripheral']['developer'])
    peripheral_in_tensorflow = set(common_developers_data[common_developers_data['tensorflow_role'] == 'peripheral']['developer'])
    
    core_in_both = core_in_pytorch & core_in_tensorflow
    
    
    venn_dict = {
        'Peripheral developers in PyTorch': peripheral_in_pytorch,
        'Core developers in PyTorch': core_in_pytorch,
        'Core developers in TensorFlow': core_in_tensorflow,
        'Peripheral developers in TensorFlow': peripheral_in_tensorflow,
    }
    
    plot = venn.venn(venn_dict)
    fig = plot.get_figure()
    fig.savefig(f'{save_path}venn_graph.pdf', dpi=fig.dpi, bbox_inches='tight')

def spatial_analysis_of_common_developers(
    common_developers_data: pd.DataFrame, type: str
):
    """
    analyze the titles and discussions of pull requests to understand the contributions developers made

    Parameters:

    Returns:

    """

    pass


def temporal_analysis_of_common_developers(
    common_developers_data: pd.DataFrame, type: str
):
    """
    extracted and compared the time period when a developer submitted pull requests to a code repository
    """


def get_full_demographic_core_developers_data(common_developers_data: pd.DataFrame) -> pd.DataFrame:
    '''
    generate full demographic of core developers in both communities
    
    Returns:
    a dataframe with columns['developer_login', 'first_commit', 'core developer since release', 'commit count', 'LOC', 'issue count', 'degree centrality']
    '''
    pass

def analysis_common_developer(common_developers_data: pd.DataFrame):
    # both core

    # pytorch core, tensorflow peri

    # tensorflow core, pytorch peri

    # both peri

    pass
