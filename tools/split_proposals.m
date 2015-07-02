COMPCAR_ROOT = '/home/lhy/Documents/Data/CompCars';
% list_file = [COMPCAR_ROOT '/train_test_split/classification/test.txt'];
proposal_file = [COMPCAR_ROOT '/proposals/classification/train.mat'];

proposals = load(proposal_file);

f_ids = [];
r_ids = [];
s_ids = [];
fs_ids = [];
rs_ids = [];
u_ids = [];

for i=1:numel(proposals.images)
    img_path = proposals.images{i};
    lb_path = strrep(img_path, 'jpg', 'txt');
    lb_fd = fopen([COMPCAR_ROOT '/label/' lb_path]);
    l = fgetl(lb_fd);
    assert(~isempty(l));
    vp = str2num(l);
    switch vp
    case -1
        u_ids = [u_ids, i];
    case 1
        f_ids = [f_ids, i];
    case 2
        r_ids = [r_ids, i];
    case 3
        s_ids = [s_ids, i];
    case 4
        fs_ids = [fs_ids, i];
    case 5
        rs_ids = [rs_ids, i];
    end
    fclose(lb_fd);
end

images = proposals.images(f_ids);
boxes = proposals.boxes(f_ids);
save(strrep(proposal_file, '.mat', '_front.mat'), 'images', 'boxes');

images = proposals.images(r_ids);
boxes = proposals.boxes(r_ids);
save(strrep(proposal_file, '.mat', '_rear.mat'), 'images', 'boxes');

images = proposals.images(s_ids);
boxes = proposals.boxes(s_ids);
save(strrep(proposal_file, '.mat', '_side.mat'), 'images', 'boxes');

images = proposals.images(fs_ids);
boxes = proposals.boxes(fs_ids);
save(strrep(proposal_file, '.mat', '_front_side.mat'), 'images', 'boxes');

images = proposals.images(rs_ids);
boxes = proposals.boxes(rs_ids);
save(strrep(proposal_file, '.mat', '_rear_side.mat'), 'images', 'boxes');
