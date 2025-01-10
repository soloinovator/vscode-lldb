import os
import json
import io
import time
from openai import OpenAI
from openai.types.beta.assistant_stream_event import ThreadRunRequiresAction, ThreadMessageCompleted
from octokit import Octokit

class IssueAnalyzer:
    def __init__(self):
        self.octokit = Octokit()
        self.openai = OpenAI()
        self.repo_full_name = os.getenv('GITHUB_REPOSITORY')

    def handle_event(self):

        with open(os.getenv('GITHUB_EVENT_PATH'), 'rb') as f:
            event = json.load(f)

        match os.getenv('GITHUB_EVENT_NAME'):
            case 'issues':
                issue = event['issue']
            case 'workflow_dispatch':
                issue_number = int(event['inputs']['issue'])
                owner, repo = self.repo_full_name.split('/')
                response = self.octokit.issues.get(owner=owner, repo=repo, issue_number=issue_number)
                issue = response.json

        assistant = self.openai.beta.assistants.retrieve(os.getenv('ASSISTANT_ID'))

        issue_file = self.openai.files.create(
            file=('NEW_ISSUE.md', self.make_issue_content(issue)),
            purpose='assistants'
        )

        thread = self.openai.beta.threads.create(
            metadata={'issue': f'{issue["number"]}: {issue["title"]}'},
            messages=[{
                    'role': 'user',
                    'content': 'We have a new issue report, see NEW_ISSUE.md',
                    'attachments': [{
                        'file_id': issue_file.id,
                        'tools': [{'type': 'file_search'}]
                    }]
                }
            ]
        )

        thread_vstore_id = thread.tool_resources.file_search.vector_store_ids[0]
        self.wait_vector_store(thread_vstore_id)

        stream = self.openai.beta.threads.runs.create(
            assistant_id=assistant.id,
            thread_id=thread.id,
            stream=True
        )

        streams = [stream]
        while streams:
            stream = streams.pop(0)
            for event in stream:
                match event:
                    case ThreadMessageCompleted():
                        for c in event.data.content:
                            print('Assistant:', c.text.value)
                    case ThreadRunRequiresAction():
                        tool_outputs = []
                        for tool in event.data.required_action.submit_tool_outputs.tool_calls:
                            if tool.function.name == 'add_issue_labels':
                                args = json.loads(tool.function.arguments)
                                print('Tool call: add_issue_labels', args)
                                tool_outputs.append({'tool_call_id': tool.id, 'output': 'Ok'})
                            elif tool.function.name == 'set_issue_title':
                                args = json.loads(tool.function.arguments)
                                print('Tool call: set_issue_title', args)
                                tool_outputs.append({'tool_call_id': tool.id, 'output': 'Ok'})
                            elif tool.function.name == 'search_github':
                                args = json.loads(tool.function.arguments)
                                print('Tool call: search_github', args)
                                query = f'repo:{self.repo_full_name} {args["query"]}'
                                output = self.search_github(query, thread_vstore_id, exclude=[issue['number']])
                                tool_outputs.append({'tool_call_id': tool.id, 'output': output})

                        new_stream = self.openai.beta.threads.runs.submit_tool_outputs(
                            thread_id=thread.id,
                            run_id=event.data.id,
                            tool_outputs=tool_outputs,
                            stream=True)
                        streams.append(new_stream)

    def search_github(self, query: str, vstore_id: str, exclude:list=[], max_results=5) -> str:
        response = self.octokit.search.issues(q=query)
        if response.json.get('status'):
            return f'Search failed: {response.json["message"]}'

        result_lines = []
        for issue in response.json['items']:
            issue_number = issue['number']
            if issue_number in exclude:
                continue
            issue_file = self.openai.files.create(
                file=(f'ISSUE_{issue_number}.md', self.make_issue_content(issue, fetch_comments=True)),
                purpose='assistants'
            )
            self.openai.beta.vector_stores.files.create(
                vector_store_id=vstore_id,
                file_id=issue_file.id,
            )
            result_lines.append(f'Issue number: {issue_number}, file name: {issue_file.filename}')
            if len(result_lines) >= max_results:
                break

        self.wait_vector_store(vstore_id)
        result_lines.insert(0, f'Found {len(result_lines)} issues and attached as files to this thread:')
        return '\n'.join(result_lines)

    def make_issue_content(self, issue, fetch_comments=False) -> bytes:
        f = io.StringIO()
        f.write(f'### Title: {issue["title"]}\n')
        f.write(f'### Author: {issue["user"]["login"]}\n')
        f.write(f'### State: {issue["state"]}\n')
        f.write(f'### Labels: {",".join(label["name"] for label in issue["labels"])}\n')
        f.write(f'\n{issue["body"]}\n')

        if fetch_comments:
            owner, repo = self.repo_full_name.split('/')
            comments = self.octokit.issues.list_issue_comments(
                owner=owner, repo=repo, issue_number=issue['number'])
            for comment in comments.json:
                f.write(f'### Comment by {comment["user"]["login"]}\n')
                f.write(f'\n{comment["body"]}\n')

        return f.getvalue().encode('utf-8')
    
    def wait_vector_store(self, vstore_id):
        vstore = self.openai.beta.vector_stores.retrieve(vstore_id)
        print(vstore)
        while vstore.status == 'in_progress':
            print('Waiting for vector store.')
            time.sleep(1)
            vstore = self.openai.beta.vector_stores.retrieve(vstore_id)
            print(vstore)


if __name__ == '__main__':
    IssueAnalyzer().handle_event()
