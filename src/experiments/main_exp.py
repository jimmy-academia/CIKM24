from solver import get_solver
from utils import *
import time

'''
Main experiment for seller revenue: 5+1 methods x 3 breeding cases
'''

def run_experiments():
    args = default_args()
    args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'main_exp'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print('>>> (main_exp.py) Main experiments for seller revenue: \n')
    msg = f'''
         {nft_project_names}
        x {Breeding_Types}
        x {New_Baseline_Methods}'''
    print(msg)
    for nft_project_name in nft_project_names:
        args.nft_project_name = nft_project_name
        for _method in Baseline_Methods:
            for _breeding in Breeding_Types:
                result_file = args.checkpoint_dir / f'{nft_project_name}_{_method}_{_breeding}.pth'
            
                dd = 3 if _method == 'BANTER' else 2
                if result_file.exists():
                    print(f'|> {result_file} exists <|')
                    print(f'seller_revenue {torch.load(result_file)['seller_revenue'].item()} buyer_utilities {torch.load(result_file)['buyer_utilities'][:, :dd].sum(1).mean().item()}')
                else:
                    print(f'running [{nft_project_name}, {_method}, {_breeding}] experiment...')
                    args.breeding_type = _breeding
                    Solver = get_solver(args, _method)
                    start_time = time.time()
                    Solver.solve() # recommend pricing and NFT purchase for each user
                    runtime = time.time() - start_time
                    Solver.evaluate() # evaluate buyer utility, seller revenue
                    Result = {
                        'runtime': runtime,
                        'seller_revenue': Solver.seller_revenue,
                        'buyer_utilities': Solver.buyer_utilities, 
                        'pricing': Solver.pricing, 
                        'holdings': Solver.holdings, 
                        'buyer_budgets': Solver.buyer_budgets,
                        'nft_counts': Solver.nft_counts,
                    }
                    torch.save(deep_to_cpu(Result), result_file)
                    print(f'seller_revenue {Solver.seller_revenue.item()} buyer_utilities {Solver.buyer_utilities[:, :dd].sum(1).mean().item()}')
                    print('______________________________________experiment done.')

