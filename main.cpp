#include <bits/stdc++.h>
using namespace std;
#define lli long long int
#define llu unsigned long long int
#define ld long double
#define nl "\n"
#define fastinput ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
//---------------------------------------------------------------///
#define INF INT_MAX
class JobSequencing
{
public:
	char id;
	int dead;
	int profit;
	bool static comparison(JobSequencing a, JobSequencing b)
	{
		return (a.profit > b.profit);
	}
	void scheduling(JobSequencing arr[], int n)
	{
		sort(arr, arr + n, comparison);
		int result[n];
		bool slot[n];
		for (int i = 0; i < n; i++)
		{
			slot[i] = false;
		}
		for (int i = 0; i < n; i++)
		{
			for (int j = min(n, arr[i].dead) - 1; j >= 0; j--)
			{
				if (slot[j] == false)
				{
					result[j] = i;
					slot[j] = true;
					break;
				}
			}
		}
		cout << "Ans: ";
		for (int i = 0; i < n; i++)
		{
			if (slot[i])
			{
				cout << arr[result[i]].id << " ";
			}
		}
		cout << nl;
	}
};
int cntWays(int i,int j,vector<vector<int>>&arr,vector<vector<int>>&dp) {
	if(i >= 0 && j >= 0 &&  arr[i][j] == 1) {
		return 0;
	}
	if(i == 0 && j == 0) {
		return 1;
	}
	if(i < 0 || j < 0) {
		return 0;
	}
	if(dp[i][j] != -1) {
		return dp[i][j];
	}
	int up = cntWays(i - 1,j,arr,dp);
	int left = cntWays(i,j - 1,arr,dp);
	return dp[i][j] = up + left;
}
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
	int m = obstacleGrid.size();
	int n = obstacleGrid[0].size();
	vector<vector<int>>dp(m,vector<int>(n,-1));
	return cntWays(m -1,n - 1,obstacleGrid,dp);    
}
int countWayOne(int i, int j, vector<vector<int>> &dp)
{
	if (i == 0 || j == 0)
	{
		return 1;
	}
	if (dp[i][j] != -1)
		return dp[i][j];
	int up = countWayOne(i - 1, j, dp);
	int left = countWayOne(i, j - 1, dp);
	return dp[i][j] = up + left;
}
int uniquePathsOne(int m, int n)
{
	vector<vector<int>> dp(m, vector<int>(n, -1));
	return countWayOne(m - 1, n - 1, dp);
}
int longestCommonSubsequence(string text1, string text2)
{
	int m = text1.size();
	int n = text2.size();
	vector<vector<int>> dp(m + 1, vector<int>(n + 1));
	for (int i = 1; i <= m; i++)
	{
		for (int j = 1; j <= n; j++)
		{
			if (text1[i - 1] == text2[j - 1])
			{
				dp[i][j] = dp[i - 1][j - 1] + 1;
			}
			else
			{
				dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
			}
		}
	}
	return dp[m][n];
}
int fun7(int ind, int k, vector<int> &arr, vector<int> &dp)
{
	if (ind == 0)
		return 0;
	if (dp[ind] != -1)
		return dp[ind];
	int minStep = INT_MAX;
	for (int j = 1; j <= k; j++)
	{
		if (ind - j >= 0)
		{
			int jump = fun7(ind - j, k, arr, dp) + abs(arr[ind] - arr[ind - j]);
			minStep = min(jump, minStep);
		}
	}
	return dp[ind] = minStep;
}
int frogJumpWithKDistances(int n, int k, vector<int> &arr)
{
	vector<int> dp(n, -1);
	return fun7(n - 1, k, arr, dp);
}
int frogJump(int ind, vector<int> &nums, vector<int> &dp)
{
	if (ind == 0)
		return 0;
	if (dp[ind] != -1)
		return dp[ind];
	int jumpOne, jumpTwo = INT_MAX;
	jumpOne = frogJump(ind - 1, nums, dp) + abs(nums[ind] - nums[ind - 1]);
	if (ind > 1)
	{
		jumpTwo = frogJump(ind - 2, nums, dp) + abs(nums[ind] - nums[ind - 2]);
	}
	return dp[ind] = min(jumpOne, jumpTwo);
}
long countWaysToMakeChangeUtil(vector<int> &arr, int ind, int T, vector<vector<long>> &dp)
{

	if (ind == 0)
	{
		return (T % arr[0] == 0);
	}
	if (dp[ind][T] != -1)
		return dp[ind][T];
	long notTaken = countWaysToMakeChangeUtil(arr, ind - 1, T, dp);
	long taken = 0;
	if (arr[ind] <= T)
		taken = countWaysToMakeChangeUtil(arr, ind, T - arr[ind], dp);
	return dp[ind][T] = notTaken + taken;
}
int fun9(int i, int j, vector<vector<int>> &grid, vector<vector<int>> &dp)
{
	if (i == 0 && j == 0)
	{
		return grid[0][0];
	}
	if (i < 0 || j < 0)
	{
		return 1e9;
	}
	if (dp[i][j] != -1)
	{
		return dp[i][j];
	}
	int up = grid[i][j] + fun9(i - 1, j, grid, dp);
	int left = grid[i][j] + fun9(i, j - 1, grid, dp);
	return dp[i][j] = min(up, left);
}
int minPathSum9(vector<vector<int>> &grid)
{
	int m = grid.size();
	int n = grid[0].size();
	vector<vector<int>> dp(m, vector<int>(n, -1));
	return fun9(m - 1, n - 1, grid, dp);
}
long countWaysToMakeChange(vector<int> &arr, int n, int T)
{
	vector<vector<long>> dp(n, vector<long>(T + 1, -1));
	return countWaysToMakeChangeUtil(arr, n - 1, T, dp);
}
class Node2
{
public:
	int data;
	Node2 *left, *right;
};
Node2 *createNode(int data)
{
	Node2 *newNode = new Node2();
	newNode->data = data;
	newNode->left = newNode->right = NULL;
	return newNode;
}
void preorder(Node2 *root)
{
	if (root == NULL)
	{
		return;
	}
	cout << root->data << " ";
	preorder(root->left);
	preorder(root->right);
}
vector<Node2 *> constructBST(int start, int end)
{
	vector<Node2 *> tree;
	if (start > end)
	{
		tree.push_back(NULL);
		return tree;
	}
	for (int i = start; i <= end; i++)
	{
		vector<Node2 *> leftSubTree = constructBST(start, i - 1);
		vector<Node2 *> rightSubTree = constructBST(i + 1, end);
		for (int j = 0; j < leftSubTree.size(); j++)
		{
			Node2 *left = leftSubTree[j];
			for (int k = 0; k < rightSubTree.size(); k++)
			{
				Node2 *right = rightSubTree[k];
				Node2 *temp = createNode(i);
				temp->left = left;
				temp->right = right;
				tree.push_back(temp);
			}
		}
	}
	return tree;
}
vector<Node2 *> generateTrees(int n)
{
	vector<Node2 *> ans = constructBST(1, n);
	return ans;
}
bool getPath(Node2 *root, vector<Node2 *> &ans, Node2 *x)
{
	if (root == NULL)
		return false;
	ans.push_back(root);
	if (root == x)
	{
		return true;
	}
	if (getPath(root->left, ans, x) || getPath(root->right, ans, x))
	{
		return true;
	}
	ans.pop_back();
	return false;
}
Node2 *lowestCommonAncestor(Node2 *root, Node2 *p, Node2 *q)
{
	vector<Node2 *> arr1, arr2;
	if (getPath(root, arr1, p) == false || getPath(root, arr2, q) == false)
	{
		return 0;
	}
	int i = 0;
	for (i = 0; i < min(arr1.size(), arr2.size()); i++)
	{
		if (arr1[i] != arr2[i])
		{
			break;
		}
	}
	return arr1[i - 1];
}
vector<int> morrisInorderTraversal(Node2 *root)
{
	vector<int> ans;
	Node2 *cur = root;
	while (cur != NULL)
	{
		if (cur->left == NULL)
		{
			ans.push_back(cur->data);
			cur = cur->right;
		}
		else
		{
			Node2 *prev = cur->left;
			while (prev->right != NULL && prev->right != cur)
			{
				prev = prev->right;
			}
			if (prev->right == NULL)
			{
				prev->right = cur;
				cur = cur->left;
			}
			else
			{
				prev->right = NULL;
				ans.push_back(cur->data);
				cur = cur->right;
			}
		}
	}
	return ans;
}
void levelOrder(Node2 *root)
{
	if (root == NULL)
		return;
	queue<Node2 *> q;
	q.push(root);
	while (q.empty() == false)
	{
		Node2 *temp = q.front();
		cout << temp->data << " ";
		q.pop();
		if (temp->left != NULL)
			q.push(temp->left);
		if (temp->right != NULL)
			q.push(temp->right);
	}
}
vector<vector<int>> zigzagLevelOrder(Node2 *root)
{
	vector<vector<int>> ans;
	if (root == NULL)
		return ans;
	queue<Node2 *> q;
	q.push(root);
	bool leftToRight = true;
	while (q.empty() == false)
	{
		int n = q.size();
		vector<int> row(n);
		for (int i = 0; i < n; i++)
		{
			Node2 *temp = q.front();
			q.pop();

			int ind = (leftToRight) ? i : (n - 1 - i);
			row[ind] = temp->data;

			if (temp->left)
			{
				q.push(temp->left);
			}
			if (temp->right)
			{
				q.push(temp->right);
			}
		}
		leftToRight = !leftToRight;
		ans.push_back(row);
	}
	return ans;
}
bool isLeaf(Node2 *root)
{
	return !root->left && !root->right;
}
void addLeftBoundary(Node2 *root, vector<int> &ans)
{
	Node2 *cur = root->left;
	while (cur)
	{
		if (!isLeaf(cur))
		{
			ans.push_back(cur->data);
		}
		if (cur->left)
		{
			cur = cur->left;
		}
		else
		{
			cur = cur->right;
		}
	}
}
void addRightBoundary(Node2 *root, vector<int> &ans)
{
	Node2 *cur = root->right;
	vector<int> tmp;
	while (cur)
	{
		if (!isLeaf(cur))
		{
			tmp.push_back(cur->data);
		}
		if (cur->right)
		{
			cur = cur->right;
		}
		else
		{
			cur = cur->left;
		}
	}
	for (int i = tmp.size() - 1; i >= 0; i--)
	{
		ans.push_back(tmp[i]);
	}
}
void addLeaves(Node2 *root, vector<int> &ans)
{
	if (isLeaf(root))
	{
		ans.push_back(root->data);
		return;
	}
	if (root->left)
	{
		addLeaves(root->left, ans);
	}
	if (root->right)
	{
		addLeaves(root->right, ans);
	}
}
vector<int> display(Node2 *root)
{
	vector<int> ans;
	if (!root)
		return ans;
	if (!isLeaf(root))
	{
		ans.push_back(root->data);
	}
	addLeftBoundary(root, ans);
	addLeaves(root, ans);
	addRightBoundary(root, ans);
	return ans;
}
class Node1
{
public:
	int data;
	Node1 *left;
	Node1 *right;
	Node1(int val)
	{
		data = val;
		left = right = NULL;
	}
};
void inorder(Node1 *temp)
{
	if (temp == NULL)
		return;
	inorder(temp->left);
	cout << temp->data << " ";
	inorder(temp->right);
}
class Solution
{
public:
	void addEdgeW(int u, int v, vector<int> adj[])
	{
		adj[u].push_back(v);
	}
	void DFS1(int node, vector<int> &vis, stack<int> &s, vector<int> adj[])
	{
		vis[node] = 1;
		for (auto it : adj[node])
		{
			if (!vis[it])
			{
				DFS1(it, vis, s, adj);
			}
		}
		s.push(node);
	}
	void revDFS1(int node, vector<int> &vis, vector<int> transpose[])
	{
		cout << node << " ";
		vis[node] = 1;
		for (auto it : transpose[node])
		{
			if (!vis[it])
			{
				revDFS1(it, vis, transpose);
			}
		}
	}
	void addEdgeW(int u, int v, int wt, vector<pair<int, int>> adj[])
	{
		adj[u].push_back({v, wt});
	}
	void findTopoSort(int node, vector<int> &vis, stack<int> &s, vector<pair<int, int>> adj[])
	{
		vis[node] = 1;
		for (auto it : adj[node])
		{
			if (!vis[it.first])
			{
				findTopoSort(it.first, vis, s, adj);
			}
		}
		s.push(node);
	}
	void shortestPath5(int src, int n, vector<pair<int, int>> adj[])
	{
		vector<int> vis(n);
		for (int i = 0; i < n; i++)
		{
			vis[i] = 0;
		}
		stack<int> s;
		for (int i = 0; i < n; i++)
		{
			if (!vis[i])
			{
				findTopoSort(i, vis, s, adj);
			}
		}
		vector<int> distance(n);
		for (int i = 0; i < n; i++)
		{
			distance[i] = 1e9;
		}
		distance[src] = 0;

		while (!s.empty())
		{
			int node = s.top();
			s.pop();
			if (distance[node] != INF)
			{
				for (auto it : adj[node])
				{
					if (distance[node] + it.second < distance[it.first])
					{
						distance[it.first] = distance[node] + it.second;
					}
				}
			}
		}
		for (int i = 0; i < n; i++)
		{
			(distance[i] == 1e9) ? cout << "INF " : cout << distance[i] << " ";
		}
	}
	void addEdgeOne(vector<int> adj[], int u, int v)
	{
		adj[u].push_back(v);
	}
	bool checkCycle3(int node, vector<int> &vis, vector<int> &dfsVis, vector<int> adj[])
	{
		vis[node] = 1;
		dfsVis[node] = 1;
		for (auto it : adj[node])
		{
			if (!vis[it])
			{
				if (checkCycle3(it, vis, dfsVis, adj))
				{
					return true;
				}
			}
			else if (dfsVis[it])
			{
				return true;
			}
		}
		dfsVis[node] = 0;
		return false;
	}
	bool isCyclic3(int n, vector<int> adj[])
	{
		vector<int> vis(n, 0);
		vector<int> dfsVis(n, 0);
		for (int i = 0; i < n; i++)
		{
			if (!vis[i])
			{
				if (checkCycle3(i, vis, dfsVis, adj))
				{
					return true;
				}
			}
		}
		return false;
	}
	bool checkForCycle4(int s, int v, vector<int> adj[], vector<int> &visited)
	{
		queue<pair<int, int>> q;
		visited[s] = true;
		q.push({s, -1});
		while (!q.empty())
		{
			int node = q.front().first;
			int par = q.front().second;
			q.pop();

			for (auto it : adj[node])
			{
				if (!visited[it])
				{
					visited[it] = true;
					q.push({it, node});
				}
				else if (par != it)
				{
					return true;
				}
			}
		}
		return false;
	}
	bool isCycle4(int v, vector<int> adj[])
	{
		vector<int> visited(v - 1, 0);
		for (int i = 1; i <= v; i++)
		{
			if (!visited[i])
			{
				if (checkForCycle4(i, v, adj, visited))
				{
					return true;
				}
			}
		}
		return false;
	}
	void addEdgeTwo(vector<int> adj[], int u, int v)
	{
		adj[u].push_back(v);
		adj[v].push_back(u);
	}
	vector<int> bfsOfGraph(int v, vector<int> adj[])
	{
		vector<int> ans;
		vector<int> vis(v + 1, 0);
		queue<int> q;
		q.push(0);
		vis[0] = 1;
		while (!q.empty())
		{
			int val = q.front();
			q.pop();
			ans.push_back(val);
			for (auto it : adj[val])
			{
				if (!vis[it])
				{
					q.push(it);
					vis[it] = 1;
				}
			}
		}
		return ans;
	}
	void dfs(int node, vector<int> &vis, vector<int> adj[], vector<int> &ans)
	{
		ans.push_back(node);
		vis[node] = 1;
		for (auto it : adj[node])
		{
			if (!vis[it])
			{
				dfs(it, vis, adj, ans);
			}
		}
	}
	vector<int> dfsOfGraph(int v, vector<int> adj[])
	{
		vector<int> ans;
		vector<int> vis(v + 1, 0);
		for (int i = 0; i < v; i++)
		{
			if (!vis[i])
			{
				dfs(i, vis, adj, ans);
			}
		}
		return ans;
	}
	vector<int> topologicalSort(int n, vector<int> adj[])
	{
		queue<int> q;
		vector<int> indegree(n, 0);
		for (int i = 0; i < n; i++)
		{
			for (auto it : adj[i])
			{
				indegree[it]++;
			}
		}
		for (int i = 0; i < n; i++)
		{
			if (indegree[i] == 0)
			{
				q.push(i);
			}
		}
		vector<int> ans;
		while (!q.empty())
		{
			int node = q.front();
			q.pop();
			ans.push_back(node);
			for (auto it : adj[node])
			{
				indegree[it]--;
				if (indegree[it] == 0)
				{
					q.push(it);
				}
			}
		}
		return ans;
	}
	vector<int> majorityElement(vector<int> &nums)
	{
		int len = nums.size(), cnt1 = 0, cnt2 = 0;
		int m1 = -1, m2 = -1, i;
		for (i = 0; i < len; i++)
		{
			if (nums[i] == m1)
			{
				cnt1++;
			}
			else if (nums[i] == m2)
			{
				cnt2++;
			}
			else if (cnt1 == 0)
			{
				m1 = nums[i];
				cnt1 = 1;
			}
			else if (cnt2 == 0)
			{
				m2 = nums[i];
				cnt2 = 1;
			}
			else
			{
				cnt1--;
				cnt2--;
			}
		}
		vector<int> ans;
		cnt1 = cnt2 = 0;
		for (i = 0; i < len; i++)
		{
			if (nums[i] == m1)
				cnt1++;
			else if (nums[i] == m2)
				cnt2++;
		}
		if (cnt1 > len / 3)
		{
			ans.push_back(m1);
		}
		if (cnt2 > len / 3)
			ans.push_back(m2);
		return ans;
	}
	int findDuplicate(vector<int> &nums)
	{
		int slow = nums[0];
		int fast = nums[0];
		do
		{
			slow = nums[slow];
			fast = nums[nums[fast]];
		} while (slow != fast);
		slow = nums[0];
		while (slow != fast)
		{
			slow = nums[slow];
			fast = nums[fast];
		}
		return fast;
	}
	void josephusProblem(int ind, int k, vector<int> &nums, int &ans)
	{
		if (nums.size() == 1)
		{
			ans = nums[0];
			return;
		}
		ind = ((ind + k) % nums.size());
		nums.erase(nums.begin() + ind);
		josephusProblem(ind, k, nums, ans);
	}
	int findTheWinner(int n, int k)
	{
		vector<int> nums;
		int ans = -1;
		for (int i = 1; i <= n; i++)
		{
			nums.push_back(i);
		}
		k = k - 1;
		josephusProblem(0, k, nums, ans);
		return ans;
	}
	void generateSolve(int open, int close, string s, vector<string> &ans)
	{
		if (open == 0 && close == 0)
		{
			ans.push_back(s);
			return;
		}
		if (open > 0)
		{
			generateSolve(open - 1, close, s + '(', ans);
		}
		if (close > open)
		{
			generateSolve(open, close - 1, s + ')', ans);
			return;
		}
	}
	vector<string> generateParenthesis(int n)
	{
		vector<string> ans;
		string s;
		int open = n;
		int close = n;
		generateSolve(open, close, s, ans);
		return ans;
	}
	void TOH(int n, int a, int b, int c)
	{
		if (n > 0)
		{
			TOH(n - 1, a, c, b);
			cout << a << " " << c << nl;
			TOH(n - 1, b, a, c);
		}
	}
	int trap(vector<int> &arr)
	{
		int ans = 0;
		int maxLeft = 0, maxRight = 0;
		int low = 0, high = arr.size() - 1;
		while (low <= high)
		{
			if (arr[low] < arr[high])
			{
				if (arr[low] > maxLeft)
				{
					maxLeft = arr[low];
				}
				else
				{
					ans += maxLeft - arr[low];
				}
				low++;
			}
			else
			{
				if (arr[high] > maxRight)
				{
					maxRight = arr[high];
				}
				else
				{
					ans += maxRight - arr[high];
				}
				high--;
			}
		}
		return ans;
	}
	bool isSafe(int row, int col, int n, vector<string> &board)
	{
		int duprow = row;
		int dupcol = col;
		while (row >= 0 && col >= 0)
		{
			if (board[row][col] == 'Q')
			{
				return false;
			}
			row--;
			col--;
		}
		row = duprow;
		col = dupcol;
		while (col >= 0)
		{
			if (board[row][col] == 'Q')
			{
				return false;
			}
			col--;
		}
		row = duprow;
		col = dupcol;
		while (row < n && col >= 0)
		{
			if (board[row][col] == 'Q')
			{
				return false;
			}
			row++;
			col--;
		}
		return true;
	}
	void solve1(int col, int n, vector<string> &board, vector<vector<string>> &ans)
	{
		if (col == n)
		{
			ans.push_back(board);
			return;
		}
		for (int row = 0; row < n; row++)
		{
			if (isSafe(row, col, n, board))
			{
				board[row][col] = 'Q';
				solve1(col + 1, n, board, ans);
				board[row][col] = '.';
			}
		}
	}
	vector<vector<string>> solveNQueens(int n)
	{
		vector<vector<string>> ans;
		vector<string> board(n);
		string s(n, '.');
		for (int i = 0; i < n; i++)
		{
			board[i] = s;
		}
		solve1(0, n, board, ans);
		return ans;
	}
	void dispaly(int sol[4][4])
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				cout << sol[i][j] << " ";
			}
			cout << nl;
		}
	}
	bool isSafe(int maze[4][4], int x, int y)
	{
		if (x >= 0 && x < 4 && y >= 0 && y < 4 && maze[x][y] == 1)
		{
			return true;
		}
		return false;
	}
	bool solveMaze(int maze[4][4])
	{
		int sol[4][4] = {{0, 0, 0, 0},
						 {0, 0, 0, 0},
						 {0, 0, 0, 0},
						 {0, 0, 0, 0}};
		if (solveMazeUtil(maze, 0, 0, sol) == false)
		{
			cout << "Solution Doest Exist" << nl;
			return false;
		}
		dispaly(sol);
		return true;
	}
	bool solveMazeUtil(int maze[4][4], int x, int y, int sol[4][4])
	{
		if (x == 4 - 1 && y == 4 - 1 && maze[x][y] == 1)
		{
			sol[x][y] = 1;
			return true;
		}
		if (isSafe(maze, x, y) == true)
		{
			if (sol[x][y] == 1)
			{
				return false;
			}
			sol[x][y] = 1;
			if (solveMazeUtil(maze, x + 1, y, sol) == true)
			{
				return true;
			}
			if (solveMazeUtil(maze, x, y + 1, sol) == true)
			{
				return true;
			}
			sol[x][y] = 0;
			return false;
		}
		return false;
	}
	bool isSafe1(vector<vector<char>> &board, int row, int col, char c)
	{
		for (int i = 0; i < board.size(); i++)
		{
			if (board[i][col] == c)
			{
				return false;
			}
			if (board[row][i] == c)
			{
				return false;
			}
			if (board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == c)
			{
				return false;
			}
		}
		return true;
	}
	bool solveSudoku(vector<vector<char>> &board)
	{
		for (int i = 0; i < board.size(); i++)
		{
			for (int j = 0; j < board[i].size(); j++)
			{
				if (board[i][j] == '.')
				{
					for (char c = '1'; c <= '9'; c++)
					{
						if (isSafe1(board, i, j, c))
						{
							board[i][j] = c;
							if (solveSudoku(board))
							{
								return true;
							}
							else
							{
								board[i][j] = '.';
							}
						}
					}
					return false;
				}
			}
		}
		return true;
	}
	int xMoves[4] = {0, 1, 0, -1};
	int yMoves[4] = {1, 0, -1, 0};
	bool helpExist(vector<vector<char>> &board, string word, int x, int y)
	{
		if (!word.size())
		{
			return true;
		}
		if (x < 0 || x >= board.size() || y < 0 || y >= board[0].size() || board[x][y] != word[0])
		{
			return false;
		}
		char temp = board[x][y];
		board[x][y] = '*';
		for (int i = 0; i < 4; i++)
		{
			int nextX = x + xMoves[i];
			int nextY = y + yMoves[i];
			if (helpExist(board, word.substr(1), nextX, nextY))
			{
				return true;
			}
		}
		board[x][y] = temp;
		return false;
	}
	bool exist2(vector<vector<char>> &board, string &word)
	{
		for (int i = 0; i < board.size(); i++)
		{
			for (int j = 0; j < board[0].size(); j++)
			{
				if (board[i][j] == word[0])
				{
					if (helpExist(board, word, i, j))
					{
						return true;
					}
				}
			}
		}
		return false;
	}
	// From Here Series Start
	void graphAlgorithms(int n)
	{
		cout << "----------------------------------------------------------------------------------------------------------------" << nl;
		cout << "|----------------------------------------: [Welcome To Graph Question] :----------------------------------------" << nl;
		cout << "----------------------------------------------------------------------------------------------------------------" << nl;
		cout << "\n";

		switch (n)
		{
		case 1:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-: BFS Traversal :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "Breadth-first search (BFS) is an algorithm for searching a tree data structure for a node that satisfies a given property.            " << nl;
				cout << "It starts at the tree root and explores all nodes at the present depth prior to moving on to the nodes at the next depth level.       " << nl;
				cout << "Extra memory, usually a queue, is needed to keep track of the child nodes that were encountered but not yet explored.                 " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "vector<int> bfsOfGraph(int v,vector<int>adj[])  {" << nl;
				cout << "    vector<int>ans;" << nl;
				cout << "	   vector<int>vis(v+1,0);" << nl;
				cout << "    queue<int>q;" << nl;
				cout << "    q.push(0);" << nl;
				cout << "    vis[0] = 1;" << nl;
				cout << "    while(!q.empty()) {" << nl;
				cout << "        int val = q.front();" << nl;
				cout << "        q.pop();" << nl;
				cout << "        ans.push_back(val);" << nl;
				cout << "        for(auto it : adj[val]) {" << nl;
				cout << "            if(!vis[it]) {" << nl;
				cout << "                q.push(it);" << nl;
				cout << "                vis[it] = 1;" << nl;
				cout << "            }" << nl;
				cout << "       }" << nl;
				cout << "    }" << nl;
				cout << "    return bfs;" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					int t;
					cout << "\nEnter The Number of Component In Graph: ";
					cin >> t;
					while (t--)
					{
						int V, E;
						cout << "\nEnter The Total Number Vertex And Edges: ";
						cin >> V >> E;
						vector<int> adj[V];
						for (int i = 0; i < E; i++)
						{
							int u, v;
							cout << "\nEnter " << i + 1 << " Vertex and Edges: ";
							cin >> u >> v;
							adj[u].push_back(v);
							adj[v].push_back(u);
						}
						vector<int> ans = bfsOfGraph(V, adj);
						cout << "\nAns: ";
						for (int i = 0; i < ans.size(); i++)
						{
							cout << ans[i] << " ";
						}
						cout << nl << nl << nl;
					}

					cout << "You want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 2:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-: DFS Traversal :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "Depth-first search is an algorithm for traversing or searching tree or graph data structures. The algorithm starts at the root node   " << nl;
				cout << "(selecting some arbitrary node as the root node in the case of a graph) and explores as far as possible along each branch before      " << nl;
				cout << "backtracking.So the basic idea is to start from the root or any arbitrary node and mark the node and move to the adjacent unmarked    " << nl;
				cout << "node and continue this loop until there is no unmarked adjacent node. Then backtrack and check for other unmarked nodes and traverse  " << nl;
				cout << "them. Finally, print the nodes in the path.                                                                                           " << nl;
				cout << "----------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "void dfs(int node,vector<int>&vis,vector<int>adj[],vector<int>&ans) {" << nl;
				cout << "    ans.push_back(node);" << nl;
				cout << "    vis[node] = 1;" << nl;
				cout << "    for(auto it : adj[node]) {" << nl;
				cout << "        if(!vis[it]) {" << nl;
				cout << "            dfs(it,vis,adj,ans);" << nl;
				cout << "        }" << nl;
				cout << "    }" << nl;
				cout << "}" << nl;
				cout << "vector<int> dfsOfGraph(int v, vector<int>adj[]) {" << nl;
				cout << "    vector<int>ans;" << nl;
				cout << "    vector<int>vis(v+1,0);" << nl;
				cout << "    for(int i = 0; i < v; i++) {" << nl;
				cout << "        if(!vis[i]) {" << nl;
				cout << "            dfs(i,vis,adj,ans);" << nl;
				cout << "        }" << nl;
				cout << "    }" << nl;
				cout << "    return ans;" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					int t;
					cout << "\nEnter The Number of Component In Graph: ";
					cin >> t;
					while (t--)
					{
						int V, E;
						cout << "\nEnter The Total Number Vertex And Edges: ";
						cin >> V >> E;
						vector<int> adj[V];
						for (int i = 0; i < E; i++)
						{
							int u, v;
							cout << "\nEnter " << i + 1 << " Vertex and Edges: ";
							cin >> u >> v;
							adj[u].push_back(v);
							adj[v].push_back(u);
						}
						vector<int> ans = dfsOfGraph(V, adj);
						cout << "\nAns: ";
						for (int i = 0; i < ans.size(); i++)
						{
							cout << ans[i] << " ";
						}
						cout << nl << nl << nl;
					}

					cout << "You want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 3:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-: Topological Sort :->                                            |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "In computer science, a topological sort or topological ordering of a directed graph is a linear ordering of its vertices such that    " << nl;
				cout << "for every directed edge uv from vertex u to vertex v, u comes before v in the ordering. For instance, the vertices of the graph  may  " << nl;
				cout << "represent tasks to be performed, and the edges may represent constraints that one task must be performed before another; in this      " << nl;
				cout << "application, a topological ordering is just a valid sequence for the tasks. Precisely, a topological sort is a graph traversal in     " << nl;
				cout << "which each node v is visited only after all its dependencies are visited. A topological ordering is possible if and only  if the graph" << nl;
				cout << "has no directed cycles, that is, if it is a directed acyclic graph (DAG). Any DAG has at least one topological ordering,and algorithms" << nl;
				cout << "are known  for constructing a topological ordering of any DAG in linear time. Topological sorting has many applications especially in " << nl;
				cout << "ranking problems such as feedback arc set. Topological sorting is possible even when the DAG has disconnected components." << nl;
				cout << "-------------------------------------------------------:End:--------------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "void addEdge(vector<int>adj[],int u,int v) {" << nl;
				cout << "    adj[u].push_back(v);" << nl;
				cout << "}" << nl;
				cout << "vector<int> topologicalSort(int n,vector<int>adj[]) {" << nl;
				cout << "    queue<int>q;" << nl;
				cout << "    vector<int>indegree(n,0);" << nl;
				cout << "    for(int i = 0; i < n; i++) {" << nl;
				cout << "        for(auto it : adj[i]) {" << nl;
				cout << "            indegree[it]++;" << nl;
				cout << "        }" << nl;
				cout << "    }" << nl;
				cout << "    for(int i = 0; i < n; i++) {" << nl;
				cout << "        if(indegree[i] == 0) {" << nl;
				cout << "            q.push(i);" << nl;
				cout << "        }" << nl;
				cout << "    }" << nl;
				cout << "    vector<int>ans;" << nl;
				cout << "    while(!q.empty()) {" << nl;
				cout << "        int node = q.front();" << nl;
				cout << "        q.pop();" << nl;
				cout << "        ans.push_back(node);" << nl;
				cout << "        for(auto it : adj[node]) {" << nl;
				cout << "            indegree[it]--;" << nl;
				cout << "            if(indegree[it] == 0) {" << nl;
				cout << "                q.push(it);" << nl;
				cout << "            }" << nl;
				cout << "      }" << nl;
				cout << "   }" << nl;
				cout << "   return ans;" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				bool flag;
				do
				{
					int N = 6;
					vector<int> adj[N];

					addEdgeOne(adj, 5, 2);
					addEdgeOne(adj, 5, 0);
					addEdgeOne(adj, 4, 0);
					addEdgeOne(adj, 4, 1);
					addEdgeOne(adj, 3, 1);
					addEdgeOne(adj, 2, 3);

					vector<int> ans = topologicalSort(N, adj);
					cout << "\nAns: ";
					for (int i = 0; i < ans.size(); i++)
					{
						cout << ans[i] << " ";
					}
					cout << nl << nl;
					cout << "You want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 4:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                          <-:Detect Cycle In DG:->                                        |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "In Directed Graph check The Cycle." << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "bool checkCycle(int node,vector<int>&vis,vector<int>&dfsVis,vector<int>adj[]) {" << nl;
				cout << "	vis[node] = 1;" << nl;
				cout << "	dfsVis[node] = 1;" << nl;
				cout << "	for(auto it : adj[node]) {" << nl;
				cout << "		if(!vis[it]) {" << nl;
				cout << "			if(checkCycle(it,vis,dfsVis,adj)) {" << nl;
				cout << "				return true;" << nl;
				cout << "			}" << nl;
				cout << "		}" << nl;
				cout << "		else if(dfsVis[it]) {" << nl;
				cout << "			return true;" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	dfsVis[node] = 0;" << nl;
				cout << "	return false;" << nl;
				cout << "}" << nl;
				cout << "bool isCyclic(int n,vector<int>adj[]) {" << nl;
				cout << "	vector<int>vis(n,0);" << nl;
				cout << "	vector<int>dfsVis(n,0);" << nl;
				cout << "	for(int i = 0; i < n; i++) {" << nl;
				cout << "		if(!vis[i]) {" << nl;
				cout << "			if(checkCycle(i,vis,dfsVis,adj)) {" << nl;
				cout << "				return true;" << nl;
				cout << "			}" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	return false;" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nSize Of Vertex is: 6" << nl;
					int V = 6;
					vector<int> adj[V];
					addEdgeOne(adj, 0, 1);
					cout << "0,1" << nl;
					addEdgeOne(adj, 1, 2);
					cout << "1,2" << nl;
					addEdgeOne(adj, 1, 5);
					cout << "1,5" << nl;
					addEdgeOne(adj, 2, 3);
					cout << "2,3" << nl;
					addEdgeOne(adj, 3, 4);
					cout << "3,4" << nl;
					addEdgeOne(adj, 4, 0);
					cout << "4,0" << nl;
					addEdgeOne(adj, 4, 1);
					cout << "4,1" << nl;

					if (isCyclic3(V, adj))
					{
						cout << "Cycle Detected" << nl;
					}
					else
					{
						cout << "No Cycle Detected" << nl;
					}
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 5:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                          <-:Detect Cycle In UG:->                                        |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "Detect The Cycle in Undirected Graph.Using BFS" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "bool checkForCycle(int s,int v,vector<int>adj[],vector<int>&visited) {" << nl;
				cout << "	queue<pair<int,int>>q;" << nl;
				cout << "	visited[s] = true;" << nl;
				cout << "	q.push({s,-1});" << nl;
				cout << "	while(!q.empty()) {" << nl;
				cout << "		int node = q.front().first;" << nl;
				cout << "		int par = q.front().second;" << nl;
				cout << "		q.pop();" << nl;
				cout << "		for(auto it : adj[node]) {" << nl;
				cout << "			if(!visited[it]) {" << nl;
				cout << "				visited[it] = true;" << nl;
				cout << "				q.push({it,node});" << nl;
				cout << "			}" << nl;
				cout << "			else if(par != it) {" << nl;
				cout << "				return true;" << nl;
				cout << "			}" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	return false;" << nl;
				cout << "}" << nl;
				cout << "bool isCycle(int v,vector<int>adj[]) {" << nl;
				cout << "	vector<int>visited(v - 1,0);" << nl;
				cout << "	for(int i = 1; i <= v; i++) {" << nl;
				cout << "		if(!visited[i]) {" << nl;
				cout << "			if(checkForCycle(i,v,adj,visited)) {" << nl;
				cout << "				return true;" << nl;
				cout << "			}" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	return false;" << nl;
				cout << "}" << nl;
				cout << "void addEdge(vector<int>adj[],int u,int v) {" << nl;
				cout << "	adj[u].push_back(v);" << nl;
				cout << "	adj[v].push_back(u);" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "Value Of Vertex: 6" << nl;
					vector<int> adj[5];
					addEdgeTwo(adj, 0, 1);
					cout << "(0,1)" << nl;
					addEdgeTwo(adj, 0, 2);
					cout << "(0,2)" << nl;
					addEdgeTwo(adj, 2, 3);
					cout << "(2,3)" << nl;
					addEdgeTwo(adj, 1, 3);
					cout << "(1,3)" << nl;
					addEdgeTwo(adj, 2, 4);
					cout << "(2,4)" << nl;
					cout << "\nAns: ";
					if (isCycle4(5, adj))
					{
						cout << "YES" << nl;
					}
					else
					{
						cout << "NO" << nl;
					}
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 6:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                         <-:Shortest Path In DAG:->                                       |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "Find the Shortest Path in Directed Cyclic Graph.                                                                                      " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "void addEdge(int u,int v,int wt,vector<pair<int,int>>adj[]) {" << nl;
				cout << "	adj[u].push_back({v,wt});" << nl;
				cout << "}" << nl;
				cout << "void findTopoSort(int node,vector<int>&vis,stack<int>&s,vector<pair<int,int>>adj[]) {" << nl;
				cout << "	vis[node] = 1;" << nl;
				cout << "	for(auto it : adj[node]) {" << nl;
				cout << "		if(!vis[it.first]) {" << nl;
				cout << "			findTopoSort(it.first,vis,s,adj);" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	s.push(node);" << nl;
				cout << "}" << nl;
				cout << "void shortestPath(int src,int n,vector<pair<int,int>>adj[]) {" << nl;
				cout << "	vector<int>vis(n);" << nl;
				cout << "	for(int i = 0; i < n; i++) {" << nl;
				cout << "		vis[i] = 0;" << nl;
				cout << "	}" << nl;
				cout << "	stack<int>s;" << nl;
				cout << "	for(int i = 0; i < n; i++) {" << nl;
				cout << "		if(!vis[i]) {" << nl;
				cout << "			findTopoSort(i,vis,s,adj);" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	vector<int>distance(n);" << nl;
				cout << "	for(int i = 0; i < n; i++) {" << nl;
				cout << "		distance[i] = 1e9;" << nl;
				cout << "	}" << nl;
				cout << "	distance[src] = 0;" << nl;
				cout << "	while(!s.empty()) {" << nl;
				cout << "		int node = s.top();" << nl;
				cout << "		s.pop();" << nl;
				cout << "		if(distance[node] != INF) {" << nl;
				cout << "			for(auto it : adj[node]) {" << nl;
				cout << "				if(distance[node] + it.second < distance[it.first]) {" << nl;
				cout << "					distance[it.first] = distance[node] + it.second;" << nl;
				cout << "				}" << nl;
				cout << "			}" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	for(int i = 0; i < n; i++) {" << nl;
				cout << "		distance[i] == 1e9 ) ? cout<<INF :cout<<distance[i]<< ; " << nl;
				cout << "	}" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nEnter The Value Of Vertex And Edges: ";
					int V, E;
					cin >> V >> E;
					vector<pair<int, int>> adj[V];
					for (int i = 0; i < E; i++)
					{
						int u, v, wt;
						cout << "\nEnter The " << i + 1 << " Edges (u,v,wt): ";
						cin >> u >> v >> wt;
						addEdgeW(u, v, wt, adj);
					}
					int source = 0;
					cout << "\nAns: " << nl;
					shortestPath5(source, V, adj);
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 7:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                          <-:Kosarajus Algorithms:->                                      |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "Kosarajus Algorithms is used to find Strongly Connected Componenet in Graph.                                                          " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "void addEdge(int u,int v,vector<int>adj[]) {" << nl;
				cout << "	adj[u].push_back(v);" << nl;
				cout << "}" << nl;
				cout << "void DFS(int node,vector<int>&vis,stack<int>&s,vector<int>adj[]) {" << nl;
				cout << "	vis[node] = 1;" << nl;
				cout << "	for(auto it : adj[node]) {" << nl;
				cout << "		if(!vis[it]) {" << nl;
				cout << "			DFS(it,vis,s,adj);" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	s.push(node);" << nl;
				cout << "}" << nl;
				cout << "void revDFS(int node,vector<int>&vis,vector<int>transpose[]) {" << nl;
				cout << "	cout<<node<<"
						";"
					 << nl;
				cout << "	vis[node] = 1;" << nl;
				cout << "	for(auto it : transpose[node]) {" << nl;
				cout << "		if(!vis[it]) {" << nl;
				cout << "			revDFS(it,vis,transpose);" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "}" << nl;
				cout << "int main() {" << nl;
				cout << "	int V,E" << nl;
				cout << "	vector<int>adj[V + 1];" << nl;
				cout << "	addEdge(num,num,adj);" << nl;
				cout << "	stack<int>s;" << nl;
				cout << "	vector<int>vis(V + 1,0);" << nl;
				cout << "	for(int i = 1; i <= V; i++) {" << nl;
				cout << "		if(!vis[i]) {" << nl;
				cout << "			DFS(i,vis,s,adj);" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	vector<int>transpose[V + 1];" << nl;
				cout << "	for(int i = 1; i <= V; i++) {" << nl;
				cout << "		vis[i] = 0;" << nl;
				cout << "		for(auto it : adj[i]) {" << nl;
				cout << "			transpose[it].push_back(i);" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	while(!s.empty()) {" << nl;
				cout << "		int node = s.top();" << nl;
				cout << "		s.pop();" << nl;
				cout << "		if(!vis[node]) {" << nl;
				cout << "			cout<<SCC: ;" << nl;
				cout << "			p->revDFS(node,vis,transpose);" << nl;
				cout << "   		cout<<nl;" << nl;
				cout << " 		}" << nl;
				cout << " 	}" << nl;
				cout << " 	return 0;" << nl;
				cout << "} 	" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					int V = 6, E = 7;
					cout << "\nVertex: 6 And Edges: 7" << nl;
					vector<int> adj[V + 1];
					addEdgeW(1, 3, adj);
					cout << "(1,2)" << nl;
					addEdgeW(2, 1, adj);
					cout << "(2,1)" << nl;
					addEdgeW(3, 2, adj);
					cout << "(3,2)" << nl;
					addEdgeW(3, 5, adj);
					cout << "(3,5)" << nl;
					addEdgeW(4, 6, adj);
					cout << "(4,6)" << nl;
					addEdgeW(5, 4, adj);
					cout << "(5,4)" << nl;
					addEdgeW(6, 5, adj);
					cout << "(6,5)" << nl;

					stack<int> s;
					vector<int> vis(V + 1, 0);
					for (int i = 1; i <= V; i++)
					{
						if (!vis[i])
						{
							DFS1(i, vis, s, adj);
						}
					}

					vector<int> transpose[V + 1];
					for (int i = 1; i <= V; i++)
					{
						vis[i] = 0;
						for (auto it : adj[i])
						{
							transpose[it].push_back(i);
						}
					}
					cout << "\nAns: " << nl;
					while (!s.empty())
					{
						int node = s.top();
						s.pop();
						if (!vis[node])
						{
							cout << "SCC: ";
							revDFS1(node, vis, transpose);
							cout << nl;
						}
					}
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 8:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 9:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 10:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 11:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 13:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 14:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 15:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 16:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 17:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		default:
		{
			cout << "---------" << nl;
		}
			// End of Switch Case
		}
	}
	void dpAlgorithms(int n)
	{
		cout << "----------------------------------------------------------------------------------------------------------------" << nl;
		cout << "|------------------------------------------: [Welcome To DP Question] :-----------------------------------------" << nl;
		cout << "----------------------------------------------------------------------------------------------------------------" << nl;
		cout << "\n";

		switch (n)
		{
		case 18:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                           <-:Coin Change:->                                              |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "We are given an array Arr with N distinct coins and a target. We have an infinite supply of each coin denomination. We need  to find  " << nl;
				cout << "the number of ways we sum up the coin values to give us the target. Each coin can be used any number of times." << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				cout << "Time Complexity:  O(N*T)                                                                                                              " << nl;
				cout << "Space Complexity: O(N*T) + O(N)                                                                                                       " << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "long countWaysToMakeChangeUtil(vector<int>& arr,int ind, int T, vector<vector<long>>& dp) {" << nl;
				cout << "	if(ind == 0){" << nl;
				cout << "		 return (T%arr[0]==0);" << nl;
				cout << "	}" << nl;
				cout << "	if(dp[ind][T] != -1)  return dp[ind][T];" << nl;
				cout << "	long notTaken = countWaysToMakeChangeUtil(arr,ind-1,T,dp);" << nl;
				cout << "	long taken = 0;" << nl;
				cout << "	if(arr[ind] <= T) taken = countWaysToMakeChangeUtil(arr,ind,T-arr[ind],dp);" << nl;
				cout << "	return dp[ind][T] = notTaken + taken;" << nl;
				cout << "}" << nl;
				cout << "long countWaysToMakeChange(vector<int>& arr, int n, int T){" << nl;
				cout << "	vector<vector<long>> dp(n,vector<long>(T+1,-1));" << nl;
				cout << "	return countWaysToMakeChangeUtil(arr,n-1, T, dp);" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nEnter The Size Of Array: ";
					int N;
					cin >> N;
					vector<int> arr(N);
					for (int i = 0; i < N; i++)
					{
						cout << "\nEnter The " << i + 1 << " Coins: ";
						cin >> arr[i];
					}
					cout << "Enter The Target: ";
					int T;
					cin >> T;
					cout << "\nAns: " << nl;
					cout << "Total Number Of Ways: " << countWaysToMakeChange(arr, arr.size(), T) << nl;
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 19:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                             <-:Frog Jump :->                                             |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "There is a frog on the 1st step of an N stairs long staircase. The frog wants to reach the Nth stair. HEIGHT[i] is the height of the  " << nl;
				cout << "(i+1)th stair.If Frog jumps from ith to jth stair, the energy lost in the jump is given by |HEIGHT[i-1] - HEIGHT[j-1] |.In the Frog is" << nl;
				cout << "on ith staircase, he can jump either to (i+1)th stair or to (i+2)th stair. Your task is to find  the minimum total energy used by the " << nl;
				cout << "by the frog to reach from 1st stair to Nth stair.                                                                                     " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "int frogJump(int ind,vector<int>&nums,vector<int>&dp) {" << nl;
				cout << "	if(ind==0) return 0;" << nl;
				cout << "	if(dp[ind]!=-1) return dp[ind];" << nl;
				cout << "	int jumpOne,jumpTwo=INT_MAX;" << nl;
				cout << "	jumpOne=frogJump(ind-1,nums,dp)+abs(nums[ind]-nums[ind-1]);" << nl;
				cout << "	if(ind>1) {" << nl;
				cout << "		jumpTwo=frogJump(ind-2,nums,dp)+abs(nums[ind]-nums[ind-2]);" << nl;
				cout << "	}" << nl;
				cout << "	return dp[ind]=min(jumpOne,jumpTwo);" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nEnter The Size Of Array: ";
					int N;
					cin >> N;
					vector<int> arr(N);
					for (int i = 0; i < N; i++)
					{
						cout << "\nEnter The " << i + 1 << " Element: ";
						cin >> arr[i];
					}
					vector<int> dp(N, -1);
					cout << "Ans: " << frogJump(N - 1, arr, dp) << nl;
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 20:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                        <-:Frog Jump With Kth Distance:->                                 |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "There is a frog on the 1st step of an N stairs long staircase. The frog wants to reach the Nth stair. HEIGHT[i] is the  height of the " << nl;
				cout << "(i+1)th stair.If Frog jumps from ith to jth stair, the energy lost in the jump is given by |HEIGHT[i-1] - HEIGHT[j-1] |.In the Frog is" << nl;
				cout << " on ith staircase, he can jump either to (i+1)th stair or to (i+2)th stair. Your task is to find the minimum total energy used by the " << nl;
				cout << "frog to reach from 1st stair to Nth stair." << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "int fun(int ind,int k,vector<int>&arr,vector<int>&dp) {" << nl;
				cout << "	if(ind==0) return 0;" << nl;
				cout << "	if(dp[ind]!=-1) return dp[ind];" << nl;
				cout << "	int minStep=INT_MAX;" << nl;
				cout << "	for(int j=1;j<=k;j++) {" << nl;
				cout << "		if(ind-j>=0) {" << nl;
				cout << "			int jump=fun(ind-j,k,arr,dp)+abs(arr[ind]-arr[ind-j]);" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	return dp[ind]=minStep;" << nl;
				cout << "}" << nl;
				cout << "int frogJumpWithKDistances(int n,int k,vector<int>&arr) {" << nl;
				cout << "	vector<int>dp(n,-1);" << nl;
				cout << "	return fun(n-1,k,arr,dp);" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nEnter The Size Of Array: ";
					int N;
					cin >> N;
					vector<int> arr(N);
					for (int i = 0; i < N; i++)
					{
						cout << "\nEnter The " << i + 1 << " Element: ";
						cin >> arr[i];
					}
					cout << "\nEnter The K: ";
					int K;
					cin >> K;
					cout << "\nAns: " << frogJumpWithKDistances(N, K, arr) << nl;
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 21:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 22:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 23:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 24:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                         <-:Unique Path Sum I:->                                          |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]).The robot tries to move to" << nl;
				cout << "the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.Given the two   " << nl;
				cout << "integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner." << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "int countWay(int i,int j,vector<vector<int>>&dp) {" << nl;
				cout << "	if(i == 0 || j == 0) {" << nl;
				cout << "		return 1;" << nl;
				cout << "	}" << nl;
				cout << "	if(dp[i][j] != -1) return dp[i][j];" << nl;
				cout << "	int up = countWay(i- 1,j,dp);" << nl;
				cout << "	int left = countWay(i,j - 1,dp);" << nl;
				cout << "	return dp[i][j] = up + left;" << nl;
				cout << "}" << nl;
				cout << "int uniquePaths(int m, int n) {" << nl;
				cout << "	vector<vector<int>>dp(m,vector<int>(n,-1));" << nl;
				cout << "	return countWay(m - 1,n - 1,dp);" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nEnter The Size Of Matrix (M x N): ";
					int M, N;
					cin >> M >> N;
					cout << "Ans: " << uniquePathsOne(M, N) << nl;
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 25:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                        <-:Unique Path Sum II:->                                          |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "You are given an m x n integer array grid. There is a robot initially located at the top-left corner (i.e., grid[0][0]). The robot    " << nl;
				cout << "tries to move to the bottom-right corner (i.e., grid[m-1][n-1]). The robot can only move either down or right at any point in time.   " << nl;
				cout << "An obstacle and space are marked as 1 or 0 respectively in grid. A path that the robot takes cannot include any square  that is an    " << nl;
				cout << "obstacle.Return the number of possible unique paths that the robot can take to reach the bottom-right corner.                         " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "int cntWays(int i,int j,vector<vector<int>>&arr,vector<vector<int>>&dp) {" << nl;
				cout << "	if(i >= 0 && j >= 0 &&  arr[i][j] == 1) {" << nl;
				cout << "		return 0;" << nl;
				cout << "	}" << nl;
				cout << "	if(i == 0 && j == 0) {" << nl;
				cout << "		return 1;" << nl;
				cout << "	}" << nl;
				cout << "	if(i < 0 || j < 0) {" << nl;
				cout << "		return 0;" << nl;
				cout << "	}" << nl;
				cout << "	if(dp[i][j] != -1) {" << nl;
				cout << "		return dp[i][j];" << nl;
				cout << "	}" << nl;
				cout << "	int up = cntWays(i - 1,j,arr,dp);" << nl;
				cout << "	int left = cntWays(i,j - 1,arr,dp);" << nl;
				cout << "	return dp[i][j] = up + left;" << nl;
				cout << "}" << nl;
				cout << "int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {" << nl;
				cout << "	int m = obstacleGrid.size();" << nl;
				cout << "	int n = obstacleGrid[0].size();" << nl;
				cout << "	vector<vector<int>>dp(m,vector<int>(n,-1));" << nl;
				cout << "	return cntWays(m -1,n - 1,obstacleGrid,dp); " << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do {
					cout<<"\nEnter The Size Of 2D Array: ";
					int M,N;
					cin>>M>>N;
					vector<vector<int>>arr(M,vector<int>(N));
					for(int i = 0; i < M; i++) {
						cout<<"\nRow "<<i+1;
						for(int j = 0; j < N; j++) {
							cout<<"\nEnter The Column: "<<j+1<<": ";
							cin>>arr[i][j];
						}
					}
					cout<<"\nAns: "<<uniquePathsWithObstacles(arr)<<nl;
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while(flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 26:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 27:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                        <-:Minimum Path Sum:->                                            |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right,which minimizes the sum  of all numbers" << nl;
				cout << "along its path. You can only move either down or right at any point in time.                                                          " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "int fun(int i,int j,vector<vector<int>>&grid,vector<vector<int>>&dp) {" << nl;
				cout << "	if(i == 0 && j == 0) {" << nl;
				cout << "		return grid[0][0];" << nl;
				cout << "	}" << nl;
				cout << "	if(i < 0 || j < 0) {" << nl;
				cout << "		return 1e9;" << nl;
				cout << "	}" << nl;
				cout << "	if(dp[i][j] != -1) {" << nl;
				cout << "		return dp[i][j];" << nl;
				cout << "	}" << nl;
				cout << "	int up = grid[i][j] + fun(i - 1,j,grid,dp);" << nl;
				cout << "	int left = grid[i][j] + fun(i,j - 1,grid,dp);" << nl;
				cout << "	return dp[i][j] = min(up,left);" << nl;
				cout << "}" << nl;
				cout << "int minPathSum(vector<vector<int>>& grid) {" << nl;
				cout << "	int m = grid.size();" << nl;
				cout << "	int n = grid[0].size();" << nl;
				cout << "	vector<vector<int>>dp(m,vector<int>(n,-1));" << nl;
				cout << "	return fun(m - 1, n - 1,grid,dp);" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nEnter The Size Of Matrix ";
					cout << "\nEnter The Size Of M: ";
					int M;
					cin >> M;
					cout << "\nEnter The Size Of N: ";
					int N;
					cin >> N;
					vector<vector<int>> arr(M, vector<int>(N));
					for (int i = 0; i < M; i++)
					{
						cout << "Row Number: " << i + 1;
						for (int j = 0; j < N; j++)
						{
							cout << "\nEnter The Column: " << j + 1 << " Value: ";
							cin >> arr[i][j];
						}
					}
					cout << "\nAns: " << minPathSum9(arr) << nl;
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 28:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:Longest Common Subsequence               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "The longest common subsequence problem is the problem of finding the longest subsequence common to all sequences in set of sequences. " << nl;
				cout << "It differs from the longest common substring problem: unlike substrings, subsequences are not required to occupy consecutive positions" << nl;
				cout << "within the original sequences." << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "int longestCommonSubsequence(string text1, string text2) {" << nl;
				cout << "	int m=text1.size();" << nl;
				cout << "	int n=text2.size();" << nl;
				cout << "	vector<vector<int>>dp(m+1,vector<int>(n+1));" << nl;
				cout << "	for(int i=1;i<=m;i++) {" << nl;
				cout << "		for(int j=1;j<=n;j++) {" << nl;
				cout << "			if(text1[i-1]==text2[j-1]) {" << nl;
				cout << "				dp[i][j]=dp[i-1][j-1]+1;" << nl;
				cout << "			}" << nl;
				cout << "			else {" << nl;
				cout << "				dp[i][j]=max(dp[i-1][j],dp[i][j-1]);" << nl;
				cout << "			}" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	return dp[m][n];" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nEnter The First String: ";
					string S1;
					cin >> S1;
					cout << "\nEnter The Second String: ";
					string S2;
					cin >> S2;
					cout << "\nAns: " << longestCommonSubsequence(S1, S2) << nl;
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 29:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 30:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 31:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 32:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 33:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 34:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		default:
		{
			cout << "---------" << nl;
		}
			// End of Switch Case
		}
	}
	void treeAlgorithms(int n)
	{
		cout << "----------------------------------------------------------------------------------------------------------------" << nl;
		cout << "|----------------------------------------: [Welcome To Tree Question] :-----------------------------------------" << nl;
		cout << "----------------------------------------------------------------------------------------------------------------" << nl;
		cout << "\n";

		switch (n)
		{
		case 35:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                         <-:Depth First Search:->                                         |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "Search The element in Depth Wise in Tree Data Structure.                                                                              " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "class Node {" << nl;
				cout << "public:" << nl;
				cout << "	int data;" << nl;
				cout << "	Node *left;" << nl;
				cout << "	Node *right;" << nl;
				cout << "	Node (int val) {" << nl;
				cout << "		data=val;" << nl;
				cout << "		left=right=NULL;" << nl;
				cout << "	}" << nl;
				cout << "};" << nl;
				cout << "void inorder(Node *temp) {" << nl;
				cout << "	if(temp==NULL) return;" << nl;
				cout << "	inorder(temp->left);" << nl;
				cout << "	cout<<temp->data<<"
						";"
					 << nl;
				cout << "	inorder(temp->right);" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					Node1 *root = new Node1(1);
					root->left = new Node1(2);
					root->right = new Node1(3);
					root->left->left = new Node1(4);
					root->left->right = new Node1(5);
					cout << "\nInorder traversals: \n";
					inorder(root);
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 36:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                          <-:Breadth First Search:->                                      |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "In Breadth First Search, Element search level wise means breadth.                                                                     " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "void levelOrder(Node *root) {" << nl;
				cout << "	if(root==NULL) return;" << nl;
				cout << "	queue<Node*>q;" << nl;
				cout << "	q.push(root);" << nl;
				cout << "	while(q.empty()==false) {" << nl;
				cout << "		Node *temp=q.front();" << nl;
				cout << "		cout<<temp->data<<  ;" << nl;
				cout << "		q.pop();" << nl;
				cout << "		if(temp->left!=NULL)" << nl;
				cout << "			q.push(temp->left);" << nl;
				cout << "		if(temp->right!=NULL)" << nl;
				cout << "			q.push(temp->right);" << nl;
				cout << "	}" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					Node2 *root = createNode(1);
					root->left = createNode(2);
					root->right = createNode(3);
					root->left->left = createNode(4);
					root->left->right = createNode(5);
					cout << "\nAns: ";
					levelOrder(root);
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 37:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:Zig Zag Travers:->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "ZigZag is a tree traversal algorithm that traverses the nodes in each level from left to right and then from right to left.           " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "vector<vector<int>> zigzagLevelOrder(Node* root) {" << nl;
				cout << "	vector<vector<int>>ans;" << nl;
				cout << "	if(root == NULL) return ans;" << nl;
				cout << "	queue<Node*>q;" << nl;
				cout << "	q.push(root);" << nl;
				cout << "	bool leftToRight = true;" << nl;
				cout << "	while(q.empty() == false) {" << nl;
				cout << "		int n = q.size();" << nl;
				cout << "		vector<int>row(n);" << nl;
				cout << "		for(int i = 0; i < n; i++) {" << nl;
				cout << "			Node *temp = q.front();" << nl;
				cout << "			q.pop();" << nl;
				cout << "			int ind = (leftToRight) ? i : (n - 1 - i);" << nl;
				cout << "			row[ind]  = temp->data;" << nl;
				cout << "			if(temp->left) {" << nl;
				cout << "				q.push(temp->left);" << nl;
				cout << "			}" << nl;
				cout << "			if(temp->right) {" << nl;
				cout << "				q.push(temp->right);" << nl;
				cout << "			}" << nl;
				cout << "		}" << nl;
				cout << "		leftToRight =! leftToRight;" << nl;
				cout << "		ans.push_back(row);" << nl;
				cout << "	}" << nl;
				cout << "	return ans;	" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					Node2 *root = createNode(1);
					root->left = createNode(2);
					root->right = createNode(3);
					root->left->left = createNode(7);
					root->left->right = createNode(6);
					root->right->left = createNode(5);
					root->right->right = createNode(4);

					vector<vector<int>> ans = zigzagLevelOrder(root);
					for (int i = 0; i < ans.size(); i++)
					{
						for (int j = 0; j < ans[i].size(); j++)
						{
							cout << ans[i][j] << " ";
						}
						cout << nl;
					}
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 38:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                          <-:Boundary Traversal:->                                        |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "The boundary traversal of the binary tree consists of the left boundary, leaves, and right boundary without duplicate nodes as the    " << nl;
				cout << "nodes may contain duplicate values.                                                                                                   " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "bool isLeaf(Node *root) {" << nl;
				cout << "	return !root->left && !root->right;" << nl;
				cout << "}" << nl;
				cout << "void addLeftBoundary(Node *root,vector<int>&ans) {" << nl;
				cout << "	Node *cur = root->left;" << nl;
				cout << "	while(cur) {" << nl;
				cout << "		if(!isLeaf(cur)) {" << nl;
				cout << "			ans.push_back(cur->data);" << nl;
				cout << "		}" << nl;
				cout << "		if(cur->left) {" << nl;
				cout << "			cur = cur->left;" << nl;
				cout << "		}" << nl;
				cout << "		else {" << nl;
				cout << "			cur = cur->right;" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "}" << nl;
				cout << "void addRightBoundary(Node *root,vector<int>&ans) {" << nl;
				cout << "	Node *cur = root->right;" << nl;
				cout << "	vector<int>tmp;" << nl;
				cout << "	while(cur) {" << nl;
				cout << "		if(!isLeaf(cur)) {" << nl;
				cout << "			tmp.push_back(cur->data);" << nl;
				cout << "		}" << nl;
				cout << "		if(cur->right) {" << nl;
				cout << "			cur = cur->right;" << nl;
				cout << "		}" << nl;
				cout << "		else {" << nl;
				cout << "			cur = cur->left;" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	for(int i = tmp.size() - 1; i >= 0; i--) {" << nl;
				cout << "		ans.push_back(tmp[i]);" << nl;
				cout << "	}" << nl;
				cout << "}" << nl;
				cout << "void addLeaves(Node *root,vector<int>&ans) {" << nl;
				cout << "	if(isLeaf(root)) {" << nl;
				cout << "		ans.push_back(root->data);" << nl;
				cout << "		return;" << nl;
				cout << "	}" << nl;
				cout << "	if(root->left) {" << nl;
				cout << "		addLeaves(root->left,ans);" << nl;
				cout << "	}" << nl;
				cout << "	if(root->right) {" << nl;
				cout << "		addLeaves(root->right,ans);" << nl;
				cout << "	}" << nl;
				cout << "}" << nl;
				cout << "vector<int> display(Node *root) {" << nl;
				cout << "	vector<int>ans;" << nl;
				cout << "	if(!root) return ans;" << nl;
				cout << "	if(!isLeaf(root)) {" << nl;
				cout << "		ans.push_back(root->data);" << nl;
				cout << "	}" << nl;
				cout << "	addLeftBoundary(root,ans);" << nl;
				cout << "	addLeaves(root,ans);" << nl;
				cout << "	addRightBoundary(root,ans);" << nl;
				cout << "	return ans;" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					Node2 *root = createNode(1);
					root->left = createNode(2);
					root->left->left = createNode(3);
					root->left->left->right = createNode(4);
					root->left->left->right->left = createNode(5);
					root->left->left->right->right = createNode(6);
					root->right = createNode(7);
					root->right->right = createNode(8);
					root->right->right->left = createNode(9);
					root->right->right->left->left = createNode(10);
					root->right->right->left->right = createNode(11);

					vector<int> ans = display(root);
					cout << "\nAns: " << nl;
					for (int i = 0; i < ans.size(); i++)
					{
						cout << ans[i] << " ";
					}
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 39:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                          <-:Morris Traversal:->                                          |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "Morris  traversal is a tree traversal algorithm that does not employ the use of recursion or a stack. In this traversal, links are    " << nl;
				cout << "created as successors and nodes are printed using these links. Finally, the changes are reverted back to restore the original tree.   " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "vector<int> morrisInorderTraversal(Node *root) {" << nl;
				cout << "	vector<int>ans;" << nl;
				cout << "	Node *cur = root;" << nl;
				cout << "	while(cur != NULL) {" << nl;
				cout << "		if(cur->left == NULL) {" << nl;
				cout << "			ans.push_back(cur->data);" << nl;
				cout << "			cur = cur->right;" << nl;
				cout << "		}" << nl;
				cout << "		else {" << nl;
				cout << "			Node *prev = cur->left;" << nl;
				cout << "			while(prev->right != NULL && prev->right != cur) {" << nl;
				cout << "				prev = prev->right;" << nl;
				cout << "			}" << nl;
				cout << "			if(prev->right == NULL) {" << nl;
				cout << "				prev->right = cur;" << nl;
				cout << "				cur = cur->left;" << nl;
				cout << "			}" << nl;
				cout << "			else {" << nl;
				cout << "				prev->right = NULL;" << nl;
				cout << "				ans.push_back(cur->data);" << nl;
				cout << "				cur = cur->right;" << nl;
				cout << "			}" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	return ans;" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					Node2 *root = createNode(1);
					root->left = createNode(2);
					root->right = createNode(3);
					root->left->left = createNode(4);
					root->left->right = createNode(5);
					root->left->right->right = createNode(6);
					vector<int> ans = morrisInorderTraversal(root);
					cout << "Ans: " << nl;
					for (int i = 0; i < ans.size(); i++)
					{
						cout << ans[i] << " ";
					}
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 40:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                         <-:Lowest Common Ancestor:->                                     |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where   " << nl;
				cout << "we allow a node to be a descendant of itself)" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "bool getPath(Node *root,vector<Node*>&ans,Node *x) {" << nl;
				cout << "	if(root == NULL) return false;" << nl;
				cout << "	ans.push_back(root);" << nl;
				cout << "	if(root == x) {" << nl;
				cout << "		return true;" << nl;
				cout << "	}" << nl;
				cout << "	if(getPath(root->left,ans,x) || getPath(root->right,ans,x)) {" << nl;
				cout << "		return true;" << nl;
				cout << "	}" << nl;
				cout << "	ans.pop_back();" << nl;
				cout << "	return false;" << nl;
				cout << "}" << nl;
				cout << "Node* lowestCommonAncestor(Node *root,Node *p,Node *q) {" << nl;
				cout << "	vector<Node*>arr1,arr2;" << nl;
				cout << "	if(getPath(root,arr1,p) == false || getPath(root,arr2,q) == false) {" << nl;
				cout << "		return 0;" << nl;
				cout << "	}" << nl;
				cout << "	int i = 0;" << nl;
				cout << "	for(i = 0; i < min(arr1.size(),arr2.size()); i++) {" << nl;
				cout << "		if(arr1[i] != arr2[i]) {" << nl;
				cout << "			break;" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	return arr1[i - 1];" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					Node2 *root = createNode(1);
					root->left = createNode(2);
					root->right = createNode(3);
					root->left->left = createNode(4);
					root->left->right = createNode(5);
					root->right->left = createNode(6);
					root->right->right = createNode(7);
					Node2 *p = root->left->right;
					Node2 *q = root->right->right;

					cout << "Ans: ";
					cout << lowestCommonAncestor(root, p, q)->data;
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 41:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                            <-:Catalan Number:->                                          |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "The number of binary search trees that will be formed with N keys can be calculated by simply evaluating the corresponding number     " << nl;
				cout << "in Catalan Number series." << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "vector<Node*> constructBST(int start,int end) {" << nl;
				cout << "	vector<Node*>tree;" << nl;
				cout << "	if(start > end) {" << nl;
				cout << "		tree.push_back(NULL);" << nl;
				cout << "		return tree;" << nl;
				cout << "	}" << nl;
				cout << "	for(int i = start; i <= end; i++) {" << nl;
				cout << "		vector<Node*>leftSubTree = constructBST(start,i - 1);" << nl;
				cout << "		vector<Node*>rightSubTree = constructBST(i + 1,end);" << nl;
				cout << "		for(int j = 0; j < leftSubTree.size(); j++) {" << nl;
				cout << "			Node *left = leftSubTree[j];" << nl;
				cout << "			for(int k = 0; k < rightSubTree.size(); k++) {" << nl;
				cout << "				Node *right = rightSubTree[k];" << nl;
				cout << "				Node *temp = createNode(i);" << nl;
				cout << "				temp->left = left;" << nl;
				cout << "				temp->right = right;" << nl;
				cout << "				tree.push_back(temp);" << nl;
				cout << "			}" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	return tree;" << nl;
				cout << "}" << nl;
				cout << "vector<Node*> generateTrees(int n) {" << nl;
				cout << "	vector<Node*>ans = constructBST(1,n);" << nl;
				cout << "	return ans;" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					vector<Node2 *> ans = generateTrees(3);
					cout << "\nAns: " << nl;
					for (int i = 0; i < ans.size(); i++)
					{
						preorder(ans[i]);
						cout << nl;
					}
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		default:
		{
			cout << "---------" << nl;
		}
			// End of Switch Case
		}
	}
	void generalAlgorithms(int n)
	{
		cout << "----------------------------------------------------------------------------------------------------------------" << nl;
		cout << "|---------------------------------------: [Welcome To General Question] :---------------------------------------" << nl;
		cout << "----------------------------------------------------------------------------------------------------------------" << nl;
		cout << "\n";

		switch (n)
		{
		case 42:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:Josephus Using Recursion:->                                      |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "In computer science and mathematics, the Josephus problem (or Josephus permutation) is a theoretical problem related to a certain     " << nl;
				cout << "counting-out game.A drawing for the Josephus problem sequence for 500 people and skipping value of 6. The horizontal axis is the      " << nl;
				cout << "number of the person.The vertical axis (top to bottom) is time (the number of cycle). A live person is drawn as green, a dead one is  " << nl;
				cout << "drawn as black.People are standing in a circle waiting to be executed. Counting begins at a specified point in the circle and proceeds" << nl;
				cout << " around the circle in a specified direction. After a specified number of people are skipped, the next person is executed.The procedure" << nl;
				cout << "is repeated with the remaining people, starting with the next person, going in the same direction and skipping the same number of     " << nl;
				cout << "people,until only one person remains, and is freed." << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "void josephusProblem(int ind,int k,vector<int>&nums,int &ans) {" << nl;
				cout << "    if(nums.size() == 1) {" << nl;
				cout << "        ans = nums[0];" << nl;
				cout << "        return;" << nl;
				cout << "    }" << nl;
				cout << "    ind = ((ind + k) % nums.size());" << nl;
				cout << "    nums.erase(nums.begin()+ind);" << nl;
				cout << "    josephusProblem(ind,k,nums,ans);" << nl;
				cout << "}" << nl;
				cout << "int findTheWinner(int n, int k) {" << nl;
				cout << "    vector<int>nums;" << nl;
				cout << "    int ans = -1;" << nl;
				cout << "    for(int i = 1; i <= n; i++) {" << nl;
				cout << "        nums.push_back(i);" << nl;
				cout << "    }" << nl;
				cout << "    k = k - 1;" << nl;
				cout << "    josephusProblem(0,k,nums,ans);" << nl;
				cout << "    return ans;" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "Enter the Value Of N: ";
					int N;
					cin >> N;
					cout << nl;
					cout << "Enter The Value Of K: ";
					int K;
					cin >> K;
					int ans = findTheWinner(N, K);
					cout << "\nAns: " << ans << nl;
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 43:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                         <-:Generate Parentheses:->                                       |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "Generate Parentheses means generate all the parenthes of specific size.                                                               " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "void generateSolve(int open,int close,string s,vector<string>&ans) {" << nl;
				cout << "    if(open==0 && close==0) {" << nl;
				cout << "        ans.push_back(s);" << nl;
				cout << "        return;" << nl;
				cout << "	}" << nl;
				cout << "   if(open>0) {" << nl;
				cout << "       generateSolve(open-1,close,s+'(',ans);" << nl;
				cout << "   }" << nl;
				cout << "   if(close>open) {" << nl;
				cout << "       generateSolve(open,close-1,s+')',ans);" << nl;
				cout << "		return;" << nl;
				cout << "	}" << nl;
				cout << "}" << nl;
				cout << "vector<string> generateParenthesis(int n) {" << nl;
				cout << "	vector<string>ans;" << nl;
				cout << "	string s;" << nl;
				cout << "	int open=n;" << nl;
				cout << "	int close=n;" << nl;
				cout << "	generateSolve(open,close,s,ans);" << nl;
				cout << "	return ans;" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nEnter The Value Of N: ";
					int N;
					cin >> N;
					vector<string> ans = generateParenthesis(N);
					cout << "Ans: " << nl;
					for (int i = 0; i < ans.size(); i++)
					{
						cout << ans[i] << nl;
					}
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 44:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                           <-:Tower Of Hanoi:->                                           |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "The Tower of Hanoi is a mathematical game or puzzle consisting of three rods and a number of disks of various diameters, which can    " << nl;
				cout << "slide onto any rod. The puzzle begins with the disks stacked on one rod in order of decreasing size, the smallest at the top, thus    " << nl;
				cout << "slide onto any rod. The puzzle begins with the disks stacked on one rod in order of decreasing size, the smallest at the top, thus    " << nl;
				cout << "approximating a conical shape.                                                                                                        " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "void TOH(int n,int a,int b,int c) {" << nl;
				cout << "	if(n > 0) {" << nl;
				cout << "		TOH(n - 1,a,c,b);" << nl;
				cout << "		cout<<a<<"
						"<<c<<nl;"
					 << nl;
				cout << "		TOH(n - 1,b,a,c);" << nl;
				cout << "	}" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nEnter The Value Of N: ";
					int N;
					cin >> N;
					cout << "\nEnter The Value Of A,B,C: ";
					int A, B, C;
					cin >> A >> B >> C;
					cout << "\nAns: ";
					TOH(N, A, B, C);
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 45:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                         <-:Trapping Rainwater:->                                         |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap      " << nl;
				cout << "after raining.                                                                                                                        " << nl;
				cout << " _________________________________________                                                                                            " << nl;
				cout << "|             __ __    __    __           |                                                                                           " << nl;
				cout << "| __         |__|__|__|__|  |__|      __  |                                                                                           " << nl;
				cout << "||__|   __   |__|__|__|__|  |__|   __|__| |                                                                                           " << nl;
				cout << "||__|__|__|__|__|__|__|__|__|__|__|__|__| |                                                                                           " << nl;
				cout << "|__2__0__1__0__3__3__2__3__0__3__0__1__2__|                                                                                           " << nl;
				cout << "|_________________________________________|                                                                                           " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "int trap(vector<int>&arr) {" << nl;
				cout << "	int ans=0;" << nl;
				cout << "	int maxLeft=0,maxRight=0;" << nl;
				cout << "	int low=0,high=arr.size()-1;" << nl;
				cout << "	while(low<=high) {" << nl;
				cout << "		if(arr[low]<arr[high]) {" << nl;
				cout << "			if(arr[low]>maxLeft) {" << nl;
				cout << "				maxLeft=arr[low];" << nl;
				cout << "			}" << nl;
				cout << "			else {" << nl;
				cout << "				ans+=maxLeft-arr[low];" << nl;
				cout << "			}" << nl;
				cout << "			low++;" << nl;
				cout << "		}" << nl;
				cout << "		else {" << nl;
				cout << "			if(arr[high]>maxRight) {" << nl;
				cout << "				maxRight=arr[high];" << nl;
				cout << "			}" << nl;
				cout << "			else {" << nl;
				cout << "				ans+=maxRight-arr[high];" << nl;
				cout << "			}" << nl;
				cout << "			 high--;" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	return ans;" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nEnter The Size of Array: ";
					int N;
					cin >> N;
					vector<int> nums(N);
					for (int i = 0; i < N; i++)
					{
						cout << "\nEnter The " << i + 1 << " Element"
							 << ": ";
						cin >> nums[i];
						cout << nl;
					}
					int ans = trap(nums);
					cout << "\nAns: " << ans << nl;
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 46:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                            <-:Job Sequencing:->                                          |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "Given an array of jobs where every job has a deadline and associated profit if the job is finished before the deadline. It is also    " << nl;
				cout << "given that every job takes a single unit of time, so the minimum possible deadline for any job is 1. How to maximize total profit if  " << nl;
				cout << "only one job can be scheduled at a time." << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "char id;" << nl;
				cout << "int dead;" << nl;
				cout << "int profit;" << nl;
				cout << "bool static comparison(JobSequencing a,JobSequencing b) {" << nl;
				cout << "	return (a.profit>b.profit);" << nl;
				cout << "}" << nl;
				cout << "void scheduling(JobSequencing arr[],int n) {" << nl;
				cout << "	sort(arr,arr+n,comparison);" << nl;
				cout << "	int result[n];" << nl;
				cout << "	bool slot[n];" << nl;
				cout << "	for(int i=0;i<n;i++) {" << nl;
				cout << "		slot[i]=false;" << nl;
				cout << "	}" << nl;
				cout << "	for(int i=0;i<n;i++) {" << nl;
				cout << "		for(int j=min(n,arr[i].dead)-1;j>=0;j--) {" << nl;
				cout << "			if(slot[j]==false) {" << nl;
				cout << "				result[j]=i;" << nl;
				cout << "				slot[j]=true;" << nl;
				cout << "				break;" << nl;
				cout << "			}" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	for(int i=0;i<n;i++) {" << nl;
				cout << "		if(slot[i]) {" << nl;
				cout << "			cout<<arr[result[i]].id<<"
						";"
					 << nl;
				cout << "		}" << nl;
				cout << "	|" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					JobSequencing *j = new JobSequencing();
					JobSequencing arr[] = {{'A', 2, 100},
										   {'B', 1, 19},
										   {'C', 2, 27},
										   {'D', 1, 25},
										   {'E', 3, 15}};
					int n = sizeof(arr) / sizeof(arr[0]);
					j->scheduling(arr, n);
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 47:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                           <-:N-Queen:->                                                  |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "The N Queen is the problem of placing N chess queens on an NxN chessboard so that no two queens attack each other                     " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "bool isSafe(int row,int col,int n,vector<string>&board) {" << nl;
				cout << "	int duprow=row;" << nl;
				cout << "	int dupcol=col;" << nl;
				cout << "	while(row>=0 && col>=0) {" << nl;
				cout << "		if(board[row][col]=='Q') {" << nl;
				cout << "			return false;" << nl;
				cout << "		}" << nl;
				cout << "		row--;" << nl;
				cout << "		col--;" << nl;
				cout << "	}" << nl;
				cout << "	row=duprow;" << nl;
				cout << "	col=dupcol;" << nl;
				cout << "	while(col>=0) {" << nl;
				cout << "		if(board[row][col]=='Q') {" << nl;
				cout << "			return false;" << nl;
				cout << "		}" << nl;
				cout << "		col--" << nl;
				cout << "	}" << nl;
				cout << "	row=duprow;" << nl;
				cout << "	col=dupcol;" << nl;
				cout << "	while(row<n && col>=0) {" << nl;
				cout << "		if(board[row][col]=='Q') {" << nl;
				cout << "			return false;" << nl;
				cout << "		}" << nl;
				cout << "		row++;" << nl;
				cout << "		col--" << nl;
				cout << "	}" << nl;
				cout << "	return true;" << nl;
				cout << "}" << nl;
				cout << "void solve(int col,int n,vector<string>&board,vector<vector<string>>&ans) {" << nl;
				cout << "	if(col==n) {" << nl;
				cout << "		ans.push_back(board);" << nl;
				cout << "		return;" << nl;
				cout << "	}" << nl;
				cout << "	for(int row=0;row<n;row++) {" << nl;
				cout << "		if(isSafe(row,col,n,board)) {" << nl;
				cout << "			board[row][col]='Q';" << nl;
				cout << "			solve(col+1,n,board,ans);" << nl;
				cout << "			board[row][col]='.';" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "}" << nl;
				cout << "vector<vector<string>> solveNQueens(int n) {" << nl;
				cout << "	vector<vector<string>>ans;" << nl;
				cout << "	vector<string>board(n);" << nl;
				cout << "	string s(n,'.');" << nl;
				cout << "	for(int i=0;i<n;i++) {" << nl;
				cout << "		board[i]=s;" << nl;
				cout << "	}" << nl;
				cout << "	solve(0,n,board,ans);" << nl;
				cout << "	return ans;" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nEnter The Value Of N: ";
					int N;
					cin >> N;
					vector<vector<string>> ans = solveNQueens(N);
					cout << "Ans: " << nl;
					for (int i = 0; i < ans.size(); i++)
					{
						cout << "Arrangement: " << i + 1 << nl;
						for (int j = 0; j < ans[0].size(); j++)
						{
							cout << ans[i][j] << nl;
						}
						cout << nl;
					}
					cout << "\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 48:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                           <-:Rat In A Maze:->                                            |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "A Maze is given as N*N binary matrix of blocks where source block is the upper left most block i.e., maze[0][0] and destination block " << nl;
				cout << "is  lower rightmost block i.e., maze[N-1][N-1]. A rat starts from source and has to reach the destination. The rat can move only in 2 " << nl;
				cout << " directions: forward and down." << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "bool solveMazeUtil(int maze[N][N],int x,int y,int sol[N][N]);" << nl;
				cout << "void dispaly(int sol[N][N]) {" << nl;
				cout << "	for(int i=0;i<N;i++) {" << nl;
				cout << "		for(int j=0;j<N;j++) {" << nl;
				cout << "			cout<<sol[i][j]<<"
						";"
					 << nl;
				cout << "		}" << nl;
				cout << "		cout<<nl;" << nl;
				cout << "	}" << nl;
				cout << "}" << nl;
				cout << "bool isSafe(int maze[N][N],int x,int y) {" << nl;
				cout << "	if(x>=0 && x<N && y>=0 && y<N && maze[x][y]==1) {" << nl;
				cout << "		return true;" << nl;
				cout << "	}" << nl;
				cout << "	return false;" << nl;
				cout << "}" << nl;
				cout << "bool solveMaze(int maze[N][N]) {" << nl;
				cout << "	int sol[N][N]={ {0,0,0,0}," << nl;
				cout << "					{0,0,0,0}," << nl;
				cout << "					{0,0,0,0}," << nl;
				cout << "					{0,0,0,0} };" << nl;
				cout << "	if(solveMazeUtil(maze,0,0,sol)==false) {" << nl;
				cout << "		cout<<Solution Doest Exist<<nl;" << nl;
				cout << "		return false;" << nl;
				cout << "	}" << nl;
				cout << "	dispaly(sol);" << nl;
				cout << "	return true;" << nl;
				cout << "}" << nl;
				cout << "bool solveMazeUtil(int maze[N][N],int x,int y,int sol[N][N]) {" << nl;
				cout << "	if(x==N-1 && y==N-1 && maze[x][y]==1) {" << nl;
				cout << "		sol[x][y]=1;" << nl;
				cout << "		return false;" << nl;
				cout << "	}" << nl;
				cout << "	if(isSafe(maze,x,y)==true) {" << nl;
				cout << "		if(sol[x][y]==1) {" << nl;
				cout << "			return false;" << nl;
				cout << "		}" << nl;
				cout << "		sol[x][y]=1;" << nl;
				cout << "		if(solveMazeUtil(maze,x+1,y,sol)==true) {" << nl;
				cout << "			return true;" << nl;
				cout << "		}" << nl;
				cout << "		if(solveMazeUtil(maze,x,y+1,sol)==true) {" << nl;
				cout << "			return true;" << nl;
				cout << "		}" << nl;
				cout << "		sol[x][y]=0;" << nl;
				cout << "		return false;" << nl;
				cout << "	}" << nl;
				cout << "	return false;" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				int maze[4][4] = {{1, 0, 0, 0},
								  {1, 1, 0, 1},
								  {0, 1, 0, 0},
								  {1, 1, 1, 1}};
				cout << "Ans: " << nl;
				solveMaze(maze);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 49:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                            <-:Sudoku Solver:->                                           |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "You have given 9x9 sudoku you have given some empty data fill the soduko solver" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "bool isSafe(vector<vector<char>>&board,int row,int col,char c) {" << nl;
				cout << "	for(int i=0;i<board.size();i++) {" << nl;
				cout << "		if(board[i][col]==c) {" << nl;
				cout << "			return false;" << nl;
				cout << "		}" << nl;
				cout << "		if(board[row][i]==c) {" << nl;
				cout << "			return false;" << nl;
				cout << "		}" << nl;
				cout << "		if(board[3*(row/3)+i/3][3*(col/3)+i%3]==c) {" << nl;
				cout << "			return false;" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	return true;" << nl;
				cout << "}" << nl;
				cout << "bool solveSudoku(vector<vector<char>>& board) {" << nl;
				cout << "	for(int i=0;i<board.size();i++) {" << nl;
				cout << "		for(int j=0;j<board[i].size();j++) {" << nl;
				cout << "			if(board[i][j]=='.') {" << nl;
				cout << "				for(char c='1';c<='9';c++) {" << nl;
				cout << "					if(isSafe(board,i,j,c)) {" << nl;
				cout << "						board[i][j]=c;" << nl;
				cout << "						if(solveSudoku(board)) {" << nl;
				cout << "							return true;" << nl;
				cout << "						}" << nl;
				cout << "						else {" << nl;
				cout << "							board[i][j]='.';" << nl;
				cout << "						}" << nl;
				cout << "					}" << nl;
				cout << "				}" << nl;
				cout << "				return false;" << nl;
				cout << "			}" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	return true;" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				vector<vector<char>> board{
					{'5', '3', '.', '.', '7', '.', '.', '.', '.'},
					{'6', '.', '.', '1', '9', '5', '.', '.', '.'},
					{'.', '9', '8', '.', '.', '.', '.', '6', '.'},
					{'8', '.', '.', '.', '6', '.', '.', '.', '3'},
					{'4', '.', '.', '8', '.', '3', '.', '.', '1'},
					{'7', '.', '.', '.', '2', '.', '.', '.', '6'},
					{'.', '6', '.', '.', '.', '.', '2', '8', '.'},
					{'.', '.', '.', '4', '1', '9', '.', '.', '5'},
					{'.', '.', '.', '.', '8', '.', '.', '7', '9'}};
				solveSudoku(board);
				cout << "Ans: " << nl;
				for (int i = 0; i < board.size(); i++)
				{
					for (int j = 0; j < board[i].size(); j++)
					{
						cout << board[i][j] << " ";
					}
					cout << nl;
				}
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 50:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                             <-:Word Search:->                                            |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "A word search, word find, word seek, word sleuth or mystery word puzzle is a word game that consists of the letters A word search,    " << nl;
				cout << "word find, word seek, word sleuth or mystery A word search, word find, word seek, word sleuth or mystery word puzzle is a word game   " << nl;
				cout << "that consists of the letters of words placed in a grid, which usually has a rectangular or square shape. The objective of this puzzle " << nl;
				cout << "is to find and mark all the words hidden inside the box." << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "int xMoves[4] = {0,1,0,-1};" << nl;
				cout << "int yMoves[4] = {1,0,-1,0};" << nl;
				cout << "bool helpExist(vector<vector<char>>& board,string word,int x,int y) {" << nl;
				cout << "	if(!word.size()) {" << nl;
				cout << "		return true;" << nl;
				cout << "	}" << nl;
				cout << "	if(x < 0 || x >= board.size() || y < 0 || y >= board[0].size() || board[x][y] != word[0]) {" << nl;
				cout << "		return false;" << nl;
				cout << "	}" << nl;
				cout << "	char temp = board[x][y];" << nl;
				cout << "	board[x][y]='*';" << nl;
				cout << "	for(int i = 0; i < 4; i++) {" << nl;
				cout << "		int nextX = x + xMoves[i];" << nl;
				cout << "		int nextY = y + yMoves[i];" << nl;
				cout << "		if(helpExist(board,word.substr(1),nextX,nextY)) {" << nl;
				cout << "			return true;" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	board[x][y] = temp;" << nl;
				cout << "	return false;" << nl;
				cout << "}" << nl;
				cout << "bool exist(vector<vector<char>>& board,string& word) {" << nl;
				cout << "	for(int i = 0; i < board.size(); i++) {" << nl;
				cout << "		for(int j = 0; j < board[0].size(); j++) {" << nl;
				cout << "			if(board[i][j] == word[0]) {" << nl;
				cout << "				if(helpExist(board,word,i,j)) {" << nl;
				cout << "					return true;" << nl;
				cout << "				}" << nl;
				cout << "			}" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	return false; " << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nEnter the Size Of Matrix M and N: ";
					int M, N;
					cin >> M >> N;
					vector<vector<char>> arr(M, vector<char>(N));
					for (int i = 0; i < M; i++)
					{
						cout << "Enter The Value Of: " << i + 1 << " Row" << nl;
						for (int j = 0; j < N; j++)
						{
							cout << "Enter The Value Of: " << j + 1 << " Columns: ";
							cin >> arr[i][j];
						}
					}
					cout << "\nEnter The String: ";
					string S;
					cin >> S;
					cout << "\n\nAns: ";
					if (exist2(arr, S))
					{
						cout << "String Exist" << nl;
					}
					else
					{
						cout << "String Not Exist" << nl;
					}
					cout << "\n\nYou want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		default:
		{
			cout << "---------" << nl;
		}
			// End of Switch Case
		}
	}
	void specialAlgorithms(int n)
	{
		cout << "----------------------------------------------------------------------------------------------------------------" << nl;
		cout << "|---------------------------------------: [Welcome To Special Question] :---------------------------------------" << nl;
		cout << "----------------------------------------------------------------------------------------------------------------" << nl;
		cout << "\n";

		switch (n)
		{
		case 51:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                    <-:Boyer - Moore Algorithms:->                                        |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "In computer science, the Boyer–Moore string-search algorithm is an efficient string-searching algorithm that is the standard          " << nl;
				cout << "benchmark for practical string-search literature. It was developed by Robert S. Boyer and J Strother Moore in 1977.The original paper " << nl;
				cout << "contained static tables for computing the pattern shifts without an explanation of how to produce them. The algorithm for producing   " << nl;
				cout << "the tables was published in a follow-on paper; this paper contained errors which were later corrected by Wojciech Rytter in 1980.The  " << nl;
				cout << "algorithm preprocesses the string being searched for (the pattern), but not the string being searched in (the text). It is thus       " << nl;
				cout << "well-suited for applications in which the pattern is much shorter than the text or where it persists across multiple searches.        " << nl;
				cout << "The Boyer–Moore algorithm uses information gathered during the preprocess step to skip sections of the text, resulting in a lower     " << nl;
				cout << "constant factor than many other string search algorithms. In general, the algorithm runs faster as the pattern length increases. The  " << nl;
				cout << "key features of the algorithm are to match on the tail of the pattern rather than the head, and to skip along the text in jumps of    " << nl;
				cout << "multiple characters rather than searching every single character in the text.                                                         " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "vector<int> majorityElement(vector<int>& nums) {" << nl;
				cout << "    int len=nums.size(),cnt1=0,cnt2=0;" << nl;
				cout << "    int m1=-1,m2=-1,i;" << nl;
				cout << "    for(i=0;i<len;i++) {" << nl;
				cout << "        if(nums[i]==m1) {" << nl;
				cout << "            cnt1++;" << nl;
				cout << "        }" << nl;
				cout << "        else if(nums[i]==m2) {" << nl;
				cout << "			cnt2++;" << nl;
				cout << "		}" << nl;
				cout << "		else if(cnt1==0) {" << nl;
				cout << "			m1=nums[i];" << nl;
				cout << "			cnt1=1;" << nl;
				cout << "		}" << nl;
				cout << "		else if(cnt2==0) {" << nl;
				cout << "			m2=nums[i];" << nl;
				cout << "			cnt2=1;" << nl;
				cout << "		}" << nl;
				cout << "		else {" << nl;
				cout << "			cnt1--;" << nl;
				cout << "			cnt2--;" << nl;
				cout << "		}" << nl;
				cout << "	}" << nl;
				cout << "	vector<int>ans;" << nl;
				cout << "	cnt1=cnt2=0;" << nl;
				cout << "	for(i=0;i<len;i++) {" << nl;
				cout << "	if(nums[i]==m1) cnt1++;" << nl;
				cout << "	else if(nums[i]==m2) cnt2++;" << nl;
				cout << "	}" << nl;
				cout << "	if(cnt1>len/3) {" << nl;
				cout << "		ans.push_back(m1);" << nl;
				cout << "	}" << nl;
				cout << "	if(cnt2>len/3)" << nl;
				cout << "		ans.push_back(m2);" << nl;
				cout << "	return ans;" << nl;
				cout << "   }" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nEnter The Size of Array: " << nl;
					int size;
					cin >> size;
					vector<int> nums(size);
					for (int i = 0; i < size; i++)
					{
						cout << "Enter The " << i + 1 << " Element Of Array: ";
						cin >> nums[i];
					}
					vector<int> ans = majorityElement(nums);
					cout << "\nAns: ";
					for (int i = 0; i < ans.size(); i++)
					{
						cout << ans[i] << " ";
					}
					cout << nl << nl;
					cout << "You want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 52:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 53:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 54:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 55:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 56:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 57:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 58:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 59:
		{
			cout << "Hello I am 59 Case";
			break;
		}
		case 60:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		default:
		{
		}
			// End of Switch Case
		}
	}
	void favouriteAlgorithms(int n)
	{
		cout << "----------------------------------------------------------------------------------------------------------------" << nl;
		cout << "|--------------------------------------: [Welcome To Favourite Question] :--------------------------------------" << nl;
		cout << "----------------------------------------------------------------------------------------------------------------" << nl;
		cout << "\n";

		switch (n)
		{
		case 61:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                   <-:Floyd's Tortoise and Hare Algorithms:->                             |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "Floyd's cycle-finding algorithm is a pointer algorithm that uses only two pointers, which move through the sequence at  different     " << nl;
				cout << "speeds. It is also called the tortoise and the hare algorithm, alluding to Aesop's fable of The Tortoise and the Hare.  The algorithm " << nl;
				cout << "is named after Robert W.                                                                                                              " << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "int findDuplicate(vector<int>& nums) {" << nl;
				cout << "    int slow=nums[0];" << nl;
				cout << "    int fast=nums[0];" << nl;
				cout << "    do {" << nl;
				cout << "        slow=nums[slow];" << nl;
				cout << "        fast=nums[nums[fast]];" << nl;
				cout << "    } while(slow!=fast);" << nl;
				cout << "    slow=nums[0];" << nl;
				cout << "    while(slow!=fast) {" << nl;
				cout << "    	slow=nums[slow];" << nl;
				cout << "       fast=nums[fast];" << nl;
				cout << "	}" << nl;
				cout << "	return fast;" << nl;
				cout << "}" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;
				bool flag;
				do
				{
					cout << "\nEnter The Size of Array: " << nl;
					int size;
					cin >> size;
					vector<int> nums(size);
					for (int i = 0; i < size; i++)
					{
						cout << "Enter The " << i + 1 << " Element Of Array: ";
						cin >> nums[i];
					}
					int ans = findDuplicate(nums);
					cout << "\nAns: ";
					cout << ans << nl;
					cout << nl;
					cout << "You want want to continue Same Program Then press 1: ";
					cin >> flag;
				} while (flag == true);
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 62:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 63:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 64:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 65:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 66:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 67:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 68:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 69:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		case 70:
		{
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "|||                                      <-:               :->                                               |||" << nl;
			cout << "|||----------------------------------------------------------------------------------------------------------|||" << nl;
			cout << "********************************************************" << nl;
			cout << "***   (1.) About                                     ***" << nl;
			cout << "***   (2.) Code                                      ***" << nl;
			cout << "***   (3.) Working                                   ***" << nl;
			cout << "********************************************************" << nl;
			cout << "Select Your Query: ";
			int nn;
			cin >> nn;
			cout << "\n\n";
			switch (nn)
			{
			case 1:
			{
				cout << "--------------------------------------------------------------(About)-----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:------------------------------------------------------------------" << nl;
				break;
			}
			case 2:
			{
				cout << "---------------------------------------------------------------(Code)----------------------------------------------------------------" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "" << nl;
				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			case 3:
			{
				cout << "--------------------------------------------------------------(Working)--------------------------------------------------------------" << nl;

				cout << "---------------------------------------------------------------:End:-----------------------------------------------------------------" << nl;
				break;
			}
			}
			// Inner Switch close here
			break;
		}
		default:
		{
			cout << "---------" << nl;
		}
			// End of Switch Case
		}
	}
	void secondPage()
	{
		cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
		cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *| PAGE :- 2 |* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
		cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
		cout << "* * * |||   :-> List Of Graph Algorithms            |||   :-> List Dynamic Programming             |||   :-> List Tree                     |||* * *" << nl;
		cout << "* * * |||   1.) BFS Traversal                       |||   18.) Coin Change                         |||   35.) Depth First Search           |||* * *" << nl;
		cout << "* * * |||   2.) DFS Traversal                       |||   19.) Frog Jump                           |||   36.) Breadth First Search         |||* * *" << nl;
		cout << "* * * |||   3.) Topological Sort                    |||   20.) Frog Jump With Kth Distance         |||   37.) Zig Zag Traversal            |||* * *" << nl;
		cout << "* * * |||   4.) Detect Cycle In DG                  |||   21.) Buy And Sell II                     |||   38.) Boundary Traversal           |||* * *" << nl;
		cout << "* * * |||   5.) Detect Cycle In UG                  |||   22.) Buy And Sell III                    |||   39.) Morris Traversal             |||* * *" << nl;
		cout << "* * * |||   6.) Shortest Path In DAG                |||   23.) Buy And Sell IV                     |||   40.) Lowest Common Ancestor       |||* * *" << nl;
		cout << "* * * |||   7.) Kosarajus Algorithms                |||   24.) Unique Path Sum I                   |||   41.) Catalan Number               |||* * *" << nl;
		cout << "* * * |||   8.) Dijkstra Algorithms                 |||   25.) Unique Path Sum II                  |||   :->List General                   |||* * *" << nl;
		cout << "* * * |||   9.) Prims Algorithms                    |||   26.) Unique Path Sum III                 |||   42.) Josephus Using Recursion     |||* * *" << nl;
		cout << "* * * |||   10.) Kruskal Algorithms                 |||   27.) Minimum Path Sum                    |||   43.) Generate Parentheses         |||* * *" << nl;
		cout << "* * * |||   11.) BellmanFord Algorithms             |||   28.) Longest Common Subsequence          |||   44.) Tower Of Hanoi               |||* * *" << nl;
		cout << "* * * |||   12.) FloydWarshall Algorithms           |||   29.) Longest Increasing Subsequence      |||   45.) Trapping Rainwater           |||* * *" << nl;
		cout << "* * * |||   13.) IsBipartite Graph                  |||   30.) Subset Sum Equal To Target          |||   46.) Job Sequencing               |||* * *" << nl;
		cout << "* * * |||   14.) Articulation Points In Graph       |||   31.) Counts Subset With K Sum            |||   47.) N-Queen                      |||* * *" << nl;
		cout << "* * * |||   15.) Bridge In Graph                    |||   32.) Word Break Problem                  |||   48.) Rat In A Maze                |||* * *" << nl;
		cout << "* * * |||   16.) Tarjan's Algorithms                |||   33.) Egg Dropping Problem                |||   49.) Sudoku Solver                |||* * *" << nl;
		cout << "* * * |||   17.) Ford Fulkerson Algorithms          |||   34.) Wildcard Matching                   |||   50.) Word Search                  |||* * *" << nl;
		cout << "* * * |--------------------------------------------------------------------------------------------------------------------------------------|* * *" << nl;
		cout << "* * * |--------------------------------------------------------------------------------------------------------------------------------------|* * *" << nl;
		cout << "* * * |||   :->  Special Algorithms                                       |||   :->  Favourite Algorithms                                  |||* * *" << nl;
		cout << "* * * |||   51.) Boyer - Moore Algorithms                                 |||   61.) Floyd's Tortoise and Hare Algorithms                  |||* * *" << nl;
		cout << "* * * |||   52.)                                                          |||   62.)                                                       |||* * *" << nl;
		cout << "* * * |||   53.)                                                          |||   63.)                                                       |||* * *" << nl;
		cout << "* * * |||   54.)                                                          |||   64.)                                                       |||* * *" << nl;
		cout << "* * * |||   55.)                                                          |||   65.)                                                       |||* * *" << nl;
		cout << "* * * |||   56.)                                                          |||   66.)                                                       |||* * *" << nl;
		cout << "* * * |||   57.)                                                          |||   67.)                                                       |||* * *" << nl;
		cout << "* * * |||   58.)                                                          |||   68.)                                                       |||* * *" << nl;
		cout << "* * * |||   59.)                                                          |||   69.)                                                       |||* * *" << nl;
		cout << "* * * |||   60.)                                                          |||   70.)                                                       |||* * *" << nl;
		cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
		cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
		cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
		bool flag1;
		do
		{
			cout << "SelectThe Algorithms which One You Want To Revise :- ";
			int query;
			cin >> query;
			cout << "\n\n";
			if (query >= 1 && query <= 17)
			{
				graphAlgorithms(query);
			}
			else if (query >= 18 && query <= 34)
			{
				dpAlgorithms(query);
			}
			else if (query >= 35 && query <= 41)
			{
				treeAlgorithms(query);
			}
			else if (query >= 42 && query <= 50)
			{
				generalAlgorithms(query);
			}
			else if (query >= 51 && query <= 60)
			{
				specialAlgorithms(query);
			}
			else if (query >= 61 && query <= 70)
			{
				favouriteAlgorithms(query);
			}
			else
			{
				cout << "Sorry, Choose Correct Algorithms Index !!! ";
			}
			cout << "\n\n\nFor Continue To The Revise Algorithms Then Press: 1 : ";
			cin >> flag1;
			cout << nl << nl;
			cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
			cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *| PAGE :- 2 |* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
			cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
			cout << "* * * |||   :-> List Of Graph Algorithms            |||   :-> List Dynamic Programming             |||   :-> List Tree                     |||* * *" << nl;
			cout << "* * * |||   1.) BFS Traversal                       |||   18.) Coin Change                         |||   35.) Depth First Search           |||* * *" << nl;
			cout << "* * * |||   2.) DFS Traversal                       |||   19.) Frog Jump                           |||   36.) Breadth First Search         |||* * *" << nl;
			cout << "* * * |||   3.) Topological Sort                    |||   20.) Frog Jump With Kth Distance         |||   37.) Zig Zag Traversal            |||* * *" << nl;
			cout << "* * * |||   4.) Detect Cycle In DG                  |||   21.) Buy And Sell II                     |||   38.) Boundary Traversal           |||* * *" << nl;
			cout << "* * * |||   5.) Detect Cycle In UG                  |||   22.) Buy And Sell III                    |||   39.) Morris Traversal             |||* * *" << nl;
			cout << "* * * |||   6.) Shortest Path In DAG                |||   23.) Buy And Sell IV                     |||   40.) Lowest Common Ancestor       |||* * *" << nl;
			cout << "* * * |||   7.) Kosarajus Algorithms                |||   24.) Unique Path Sum I                   |||   41.) Catalan Number               |||* * *" << nl;
			cout << "* * * |||   8.) Dijkstra Algorithms                 |||   25.) Unique Path Sum II                  |||   :->List General                   |||* * *" << nl;
			cout << "* * * |||   9.) Prims Algorithms                    |||   26.) Unique Path Sum III                 |||   42.) Josephus Using Recursion     |||* * *" << nl;
			cout << "* * * |||   10.) Kruskal Algorithms                 |||   27.) Minimum Path Sum                    |||   43.) Generate Parentheses         |||* * *" << nl;
			cout << "* * * |||   11.) BellmanFord Algorithms             |||   28.) Longest Common Subsequence          |||   44.) Tower Of Hanoi               |||* * *" << nl;
			cout << "* * * |||   12.) FloydWarshall Algorithms           |||   29.) Longest Increasing Subsequence      |||   45.) Trapping Rainwater           |||* * *" << nl;
			cout << "* * * |||   13.) IsBipartite Graph                  |||   30.) Subset Sum Equal To Target          |||   46.) Job Sequencing               |||* * *" << nl;
			cout << "* * * |||   14.) Articulation Points In Graph       |||   31.) Counts Subset With K Sum            |||   47.) N-Queen                      |||* * *" << nl;
			cout << "* * * |||   15.) Bridge In Graph                    |||   32.) Word Break Problem                  |||   48.) Rat In A Maze                |||* * *" << nl;
			cout << "* * * |||   16.) Tarjan's Algorithms                |||   33.) Egg Dropping Problem                |||   49.) Sudoku Solver                |||* * *" << nl;
			cout << "* * * |||   17.) Ford Fulkerson Algorithms          |||   34.) Wildcard Matching                   |||   50.) Word Search                  |||* * *" << nl;
			cout << "* * * |--------------------------------------------------------------------------------------------------------------------------------------|* * *" << nl;
			cout << "* * * |--------------------------------------------------------------------------------------------------------------------------------------|* * *" << nl;
			cout << "* * * |||   :->  Special Algorithms                                       |||   :->  Favourite Algorithms                                  |||* * *" << nl;
			cout << "* * * |||   51.) Boyer - Moore Algorithms                                 |||   61.) Floyd's Tortoise and Hare Algorithms                  |||* * *" << nl;
			cout << "* * * |||   52.)                                                          |||   62.)                                                       |||* * *" << nl;
			cout << "* * * |||   53.)                                                          |||   63.)                                                       |||* * *" << nl;
			cout << "* * * |||   54.)                                                          |||   64.)                                                       |||* * *" << nl;
			cout << "* * * |||   55.)                                                          |||   65.)                                                       |||* * *" << nl;
			cout << "* * * |||   56.)                                                          |||   66.)                                                       |||* * *" << nl;
			cout << "* * * |||   57.)                                                          |||   67.)                                                       |||* * *" << nl;
			cout << "* * * |||   58.)                                                          |||   68.)                                                       |||* * *" << nl;
			cout << "* * * |||   59.)                                                          |||   69.)                                                       |||* * *" << nl;
			cout << "* * * |||   60.)                                                          |||   70.)                                                       |||* * *" << nl;
			cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
			cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
			cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;

		} while (flag1 == true);
	}
	void firstPage()
	{
		cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
		cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *| PAGE :- 1 |* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
		cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
		cout << "* * *                                                                                                                                         * * *" << nl;
		cout << "* * *                                                                                                                                         * * *" << nl;
		cout << "* * *  ||             ||   ||$$$%$$$$$$$$$   ||                     @@@@#@@@@@        @@@$@@@        ||#        //||   ||####@########        * * *" << nl;
		cout << "* * *  ||             ||   ||$$$$$$$%$$$$$   ||                    @                 @       @       || #      // ||   ||#########@###        * * *" << nl;
		cout << "* * *  ||             ||   ||                ||                   @                 @         @      ||  #    //  ||   ||                     * * *" << nl;
		cout << "* * *  ||     //@     ||   ||$$$%$$$$$$$$$   ||                 @                 @             @    ||   #  //   ||   ||#########$###        * * *" << nl;
		cout << "* * *  ||    //  @    ||   ||$$$$$$$%$$$$$   ||                @                 @               @   ||    #//    ||   ||##$##########        * * *" << nl;
		cout << "* * *  ||   //    @   ||   ||                ||                 @                 @             @    ||           ||   ||                     * * *" << nl;
		cout << "* * *  ||  //      @  ||   ||                ||                   @                @           @     ||           ||   ||                     * * *" << nl;
		cout << "* * *  || //        @ ||   ||$$$$$$$%$$$$$   ||#####&&&#####        @               @         @      ||           ||   ||##%##########        * * *" << nl;
		cout << "* * *  ||//          @||   ||$%$$$$$$$$$$$   ||#############         @@#@@@@@@        @@@$@@@        ||           ||   ||########%####        * * *" << nl;
		cout << "* * *                                                                                                                                         * * *" << nl;
		cout << "* * *                                                                                                                                         * * *" << nl;
		cout << "* * *|---------------------------------------------------------------------------------------------------------------------------------------|* * *" << nl;
		cout << "* * *|-----------------------------------<-:[ Standard Algorithms Quick Revision Using System Design ]:->------------------------------------|* * *" << nl;
		cout << "* * *|---------------------------------------------------------------------------------------------------------------------------------------|* * *" << nl;
		cout << "* * *                                                                                                                                         * * *" << nl;
		cout << "* * *                                                                                                                                         * * *" << nl;
		cout << "* * *||-------------------------------------------------------------------------------------------------------------------------------------||* * *" << nl;
		cout << "* * *|:-:|                Teacher Name :-                                                 Respected Rameshwar Cambow                     |:-:|* * *" << nl;
		cout << "* * *||-------------------------------------------------------------------------------------------------------------------------------------||* * *" << nl;
		cout << "* * *|                              Course Name :-  GEN-331 ( WORKSHOP ON DESIGN THINKING FOR INNOVATION )                                   |* * *" << nl;
		cout << "* * *||-------------------------------------------------------------------------------------------------------------------------------------||* * *" << nl;
		cout << "* * *                                                                                                                                         * * *" << nl;
		cout << "* * *|---------------------------------------------------------------------------------------------------------------------------------------|* * *" << nl;
		cout << "* * *||  S.No  ||      Registration Number     ||   First Name        ||   Last Name        ||   Roll Number     ||   Section  ||   Group   ||* * *" << nl;
		cout << "* * *||--------||------------------------------||---------------------||--------------------||-------------------||------------||---------  ||* * *" << nl;
		cout << "* * *|| (1.)   ||        11910254              ||  Pankaj             ||  Kumar             ||        05         ||   KO138    ||     A     ||* * *" << nl;
		cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
		cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;
		cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << nl;

		cout << "Press Any Key For Continue ";
		string a;
		getline(cin, a);
		cout << "\n\n\n";
		if (a.size() >= 1)
		{
			secondPage();
		}
	}
};
int main()
{
	Solution *p = new Solution();
	p->firstPage();
	return 0;
}
