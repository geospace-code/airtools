function plan = buildfile
plan = buildplan(localfunctions);
plan.DefaultTasks = "test";
end

function checkTask(~)
% Identify code issues (recursively all Matlab .m files)
issues = codeIssues;
assert(isempty(issues.Issues), formattedDisplayText(issues.Issues))
end

function testTask(~)
r = runtests('airtools', strict=true);
assert(~isempty(r), "No tests were run")
assertSuccess(r)
end
