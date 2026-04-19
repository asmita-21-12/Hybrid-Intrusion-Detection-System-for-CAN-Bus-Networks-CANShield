import os
import subprocess
from datetime import datetime


def run_git_command(args, repo_path='.', capture_output=True):
    try:
        result = subprocess.run(
            ['git'] + args,
            cwd=repo_path,
            check=True,
            text=True,
            capture_output=capture_output,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() if exc.stderr else str(exc)
        raise RuntimeError(f"Git command failed: git {' '.join(args)}\n{message}")


def is_git_repository(repo_path='.'): 
    return os.path.isdir(os.path.join(repo_path, '.git'))


def has_remote_origin(repo_path='.'):
    try:
        output = run_git_command(['remote', '-v'], repo_path=repo_path)
        return 'origin' in output
    except RuntimeError:
        return False


def current_branch(repo_path='.'):
    try:
        output = run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'], repo_path=repo_path)
        return output
    except RuntimeError:
        return None


def commit_all(message=None, repo_path='.'):
    if not is_git_repository(repo_path):
        return False, 'Git repository not initialized.'

    try:
        run_git_command(['add', '.'], repo_path=repo_path)
    except RuntimeError as exc:
        return False, f'Failed to stage files: {exc}'

    commit_message = message or f"Auto backup: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"

    try:
        output = run_git_command(['commit', '-m', commit_message], repo_path=repo_path)
        return True, output
    except RuntimeError as exc:
        if 'nothing to commit' in str(exc).lower():
            return False, 'No changes to commit.'
        return False, str(exc)


def push_to_github(message=None, branch='main', repo_path='.'):
    if not is_git_repository(repo_path):
        return False, 'Git repository not initialized. Run git init first.'

    if not has_remote_origin(repo_path):
        return False, 'Remote origin not configured. Run git remote add origin <repo_url>.'

    commit_status, commit_info = commit_all(message=message, repo_path=repo_path)
    if not commit_status and 'No changes to commit' in commit_info:
        commit_info = 'No new changes to commit; continuing to push existing history.'
    elif not commit_status:
        return False, commit_info

    try:
        run_git_command(['push', 'origin', branch], repo_path=repo_path)
        return True, f'Pushed to origin/{branch}. {commit_info}'
    except RuntimeError as exc:
        text = str(exc)
        if 'could not resolve host' in text.lower() or 'unable to access' in text.lower():
            return False, 'Network appears unavailable. Push skipped safely.'
        return False, text


def create_experiment_branch(branch='experiment', repo_path='.'):
    if not is_git_repository(repo_path):
        return False, 'Git repository not initialized.'

    try:
        run_git_command(['checkout', '-B', branch], repo_path=repo_path)
        return True, f'Branch {branch} created or switched to.'
    except RuntimeError as exc:
        return False, str(exc)


def backup_after_training(repo_path='.', branch='main', message=None):
    print('Backing up after model training...')
    return push_to_github(message=message or 'Auto backup: model training complete', branch=branch, repo_path=repo_path)


def backup_after_simulation(repo_path='.', branch='main', message=None):
    print('Backing up after simulation run...')
    return push_to_github(message=message or 'Auto backup: simulation complete', branch=branch, repo_path=repo_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GitHub backup helper for the CANShield project.')
    parser.add_argument('message', nargs='?', default=None, help='Commit message for the backup.')
    parser.add_argument('--branch', default='main', help='Git branch to push to (default: main).')
    parser.add_argument('--experiment', action='store_true', help='Create and use the experiment branch.')
    args = parser.parse_args()

    if args.experiment:
        success, info = create_experiment_branch(branch=args.branch)
        print(info)
        if not success:
            raise SystemExit(1)

    success, info = push_to_github(message=args.message, branch=args.branch)
    print(info)
    if not success:
        raise SystemExit(1)
