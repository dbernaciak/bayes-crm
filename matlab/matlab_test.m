% python config
if count(py.sys.path,pwd) == 0
    insert(py.sys.path,int32(0),pwd);
end

py.importlib.import_module('crm');
py.importlib.import_module('numpy');

% freeze random seed
st = rand('state');
rand('state',0);

% generate arrival times
n = 100; % number of samples
u = cumsum(-log(rand(1, n))); % arrival times

% Beta process test
m = 1;
c = 2;
bp = py.crm.levy_processes.beta_process(m,c);
g_bp = py.crm.levy_processes.g_beta_process(m,c);
ap_beta = py.crm.crm_approx.ApproxProcess(bp, 1001, g_bp, -1, thr=0.8);


arrival_times = py.numpy.array(u);
j_beta = double(ap_beta.generate(arrival_times));

% Gamma process test
m = 1;
gp = py.crm.levy_processes.gamma_process(m);
g_gp = py.crm.levy_processes.g_gamma_process(m);
ap_gamma = py.crm.crm_approx.ApproxProcess(gp, 1001, g_gp, -1, ...
    thr=0.8, bounds=[0; py.numpy.inf]);

arrival_times = py.numpy.array(u);
j_gamma = double(ap_gamma.generate(arrival_times));

% Custom Levy intensity function
M = 1;
c = 2;
alpha = 2;
inten_f = py.eval(sprintf('lambda x: %f * %f * x ** (-1) * (1 - x) ** (%f - 1) + %f * (%f * (%f - 1) / %f) * (1 - x) ** (%f - 2)', ...
    M, c, c, M, c, c, alpha, c) , py.dict());
ap_custom = py.crm.crm_approx.ApproxProcess(inten_f, 1001, bounds=[0; 1]);

arrival_times = py.numpy.array(u);
j_custom = double(ap_custom.generate(arrival_times));