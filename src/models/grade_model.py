import torch
import torch.nn as nn

from .attention import ConditionalAttentionPooling


class GradePredictionModel(nn.Module):
    def __init__(self, inputs_description, params):
        super().__init__()
        self.educational_programme_emb_l = nn.Embedding(*inputs_description['Educational Program / Major'][1])
        self.course_emb_l = nn.Embedding(*inputs_description['Course'][1])
        self.instructor_name_emb_l = nn.Embedding(*inputs_description['Instructor Full Name'][1])
        self.instructor_department_emb_l = nn.Embedding(*inputs_description['Instructor Department'][1])

        self.encoded_dim = 0
        for name, cfg in inputs_description.items():
            t = cfg[0]
            if t == 'number':
                self.encoded_dim += 1
            elif t == 'one-hot':
                self.encoded_dim += cfg[1]
            elif t == 'embedding':
                self.encoded_dim += cfg[1][1]

        self.other_encoded_dim = self.encoded_dim + 1

        self.attn = ConditionalAttentionPooling(
            query_dim=self.encoded_dim,
            input_dim=self.other_encoded_dim,
            hidden_dim=params['attn_hidden']
        )

        self.lstm = nn.LSTM(
            input_size=params['attn_hidden'],
            hidden_size=params['lstm_hidden'],
            batch_first=True,
            dropout=params['dropout'],
            num_layers=params['num_lstm_layers']
        )

        self.fc = nn.Sequential(
            nn.Linear(params['lstm_hidden'] + self.encoded_dim, params['l1']),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(params['l1'], params['l2']),
            nn.ReLU(),
            nn.Linear(params['l2'], params['num_classes'])
        )

    def combine_features(self, tensor_features, educational_program, course, instructor_name,
                         instructor_department, **kwargs):
        emb_educational_programme = self.educational_programme_emb_l(educational_program)
        emb_course = self.course_emb_l(course)
        emb_instructor_name = self.instructor_name_emb_l(instructor_name)
        emb_instructor_department = self.instructor_department_emb_l(instructor_department)

        return torch.cat([tensor_features, emb_educational_programme, emb_course,
                          emb_instructor_name, emb_instructor_department], dim=-1)

    def forward(self, tensor_features, educational_program, course, instructor_name,
                instructor_department, other_semesters, **kwargs):
        combined_features = self.combine_features(tensor_features, educational_program, course, instructor_name,
                                                  instructor_department)
        other_semesters_combined_features = self.combine_features(**other_semesters)
        semesters_attn = self.attn(combined_features, other_semesters_combined_features, other_semesters['mask'])
        lstm_output, (h_n, c_n) = self.lstm(semesters_attn)
        c_last = c_n[-1]
        return self.fc(torch.cat([combined_features, c_last ], dim=1))
