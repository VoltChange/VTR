import argparse

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT')
    parser.add_argument('-s', '--samples', type=int, default=500)
    args = parser.parse_args()
    dataset = args.dataset
    dir = 'data/' + dataset
    rebuilt_csv = dir + '/cluster_' + str(args.samples) + '.csv'
    rebuilt_df = pd.read_csv(rebuilt_csv)
    # 数据预处理 - 全部转为小写
    rebuilt_df[['a1', 'a2', 'b1', 'b2']] = rebuilt_df[['a1', 'a2', 'b1', 'b2']].applymap(
        lambda x: str(x).lower() if isinstance(x, str) else x)
    # 使用melt函数转换数据
    df_A = rebuilt_df.melt(id_vars='video_id', value_vars=['a1', 'a2'], var_name='dummy', value_name='sentence')
    df_B = rebuilt_df.melt(id_vars='video_id', value_vars=['b1', 'b2'], var_name='dummy', value_name='sentence')
    # 删除不需要的'dummy'列
    df_A.drop('dummy', axis=1, inplace=True)
    df_B.drop('dummy', axis=1, inplace=True)
    # 保存新的CSV文件
    df_A.to_csv(dir + '/cluster_'+str(args.samples)+'_main'+'.csv', index=False)
    df_B.to_csv(dir + '/cluster_'+str(args.samples)+'_nonmain'+'.csv', index=False)
