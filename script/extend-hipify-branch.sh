#!/bin/bash

# Hipifies the current working tree and adds the changes to the tip of the
# hipified branch, using the current commit as the second parent of a merge
# commit with the working tree matching the hipified state.
#
# So when a new commit is added to hip-ready, you can run this script to achieve
# the following history change, where the working tree of commit X' is the
# result of hipifying commit X.
#
# Before:
#
# hipify-inplace    A'--B'------D'
#                  /   /       /
# hip-ready       A---B---C---D---E
#
# After:
#
# hipify-inplace    A'--B'------D'--E'
#                  /   /       /   /
# hip-ready       A---B---C---D---E


# Note that this uses low-level git commands to construct a commit with the
# working tree and parent relationships that we want as doing so with
# higher-level commands like git commit and git merge is quite tricky because
# the "smarts" of these commands actually get in the way.

set -euo pipefail

HIPIFIED_BRANCH="${1:-hipify-inplace}"

function main() {
    local hip_ready_sha="$(git rev-parse HEAD)"
    local tmp_branch="hipify-$(echo "${hip_ready_sha}" | head -c 5)"
    local message="Hipify from nod-ai/dgl@${hip_ready_sha:0:10}"
    
    # Hipify the current working tree on a new temporary branch.
    git switch -c "${tmp_branch}"
    echo "Running hipify script"
    script/hipify-inplace.sh

    # Create a commit with the hipified changes.
    echo "Committing hipified changes"
    git add -A
    git commit -m "${message}"

    git switch "${HIPIFIED_BRANCH}"

    # Here we manually construct a git commit using low-level commands. The
    # working tree is taken from the commit we just created on ${tmp_branch}
    # and the parents are specified as the current HEAD of ${HIPIFIED_BRANCH}
    # and the HEAD of the branch we are hipifying from (in that order). See
    # https://stackoverflow.com/q/48560351.
    echo "Constructing new merge commit"
    local new_commit="$(git commit-tree -m "${message}" "${tmp_branch}^{tree}" -p HEAD -p "${hip_ready_sha}")"
    echo "Resetting ${HIPIFIED_BRANCH} to new commit ${new_commit:0:10}"
    # Then we set the hipified branch to point to the new commit.
    git reset --hard "${new_commit}"
    echo "Cleaning up temporary branch ${tmp_branch}"
    git branch -D "${tmp_branch}"
    git log -n 20 --graph --oneline
}

main
